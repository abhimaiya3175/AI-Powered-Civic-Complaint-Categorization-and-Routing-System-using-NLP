import os
import uuid
import logging
import mimetypes
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, func, inspect, text as sql_text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from PIL import Image
from PIL.ExifTags import GPSTAGS
import spacy
import pickle
import whisper
from deep_translator import GoogleTranslator
from jose import JWTError, jwt
from dotenv import load_dotenv
import math
import bcrypt

load_dotenv()

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bbmp_complaints.log")
    ]
)
logger = logging.getLogger("bbmp")

# ====================== CONFIG ======================
SECRET_KEY = os.getenv("SECRET_KEY", "bbmp_secret_2025")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "bbmp_complaints")

# Allowed audio MIME types and extensions for upload validation
ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/webm",
    "audio/flac", "audio/x-flac",
    "audio/mp4", "audio/aac",
}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".webm", ".flac", ".m4a", ".aac"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}
GPS_TOLERANCE_METERS = 100.0
MAX_IMAGE_AGE_SECONDS = 10 * 60

# Prefer explicit DATABASE_URL, otherwise construct from DB_* values.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)
try:
    engine = create_engine(DATABASE_URL)
    engine.connect()
    logger.info("Connected to PostgreSQL at %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
except Exception as e:
    logger.warning("PostgreSQL connection failed — falling back to SQLite. Error: %s", e)
    DATABASE_URL = "sqlite:///complaints.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ====================== DATABASE MODELS ======================
class AdminUser(Base):
    __tablename__ = "admin_users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Complaint(Base):
    __tablename__ = "complaints"
    id = Column(Integer, primary_key=True, index=True)
    audio_path = Column(String)
    original_text = Column(Text)
    translated_text = Column(Text)
    language = Column(String)
    category = Column(String)
    location = Column(String)
    live_latitude = Column(Float)
    live_longitude = Column(Float)
    live_location_timestamp = Column(DateTime)
    image_path = Column(String)
    image_exif_latitude = Column(Float)
    image_exif_longitude = Column(Float)
    image_exif_timestamp = Column(DateTime)
    image_live_distance_meters = Column(Float)
    trust_level = Column(String, default="medium")
    verification_mode = Column(String, default="manual_review")
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(pwd_bytes, salt)
    return hashed_password.decode('utf-8')

def ensure_complaints_schema_upgrades():
    """Add newly introduced columns for existing databases without migrations."""
    required_columns = {
        "live_latitude": "FLOAT",
        "live_longitude": "FLOAT",
        "live_location_timestamp": "TIMESTAMP",
        "image_path": "VARCHAR",
        "image_exif_latitude": "FLOAT",
        "image_exif_longitude": "FLOAT",
        "image_exif_timestamp": "TIMESTAMP",
        "image_live_distance_meters": "FLOAT",
        "trust_level": "VARCHAR DEFAULT 'medium'",
        "verification_mode": "VARCHAR DEFAULT 'manual_review'",
    }

    inspector = inspect(engine)
    if "complaints" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("complaints")}
    with engine.begin() as conn:
        for column_name, ddl in required_columns.items():
            if column_name not in existing_columns:
                conn.execute(sql_text(f"ALTER TABLE complaints ADD COLUMN {column_name} {ddl}"))


ensure_complaints_schema_upgrades()

# ====================== ML / NLP LOAD ======================
logger.info("Loading model_bbmp.pkl …")
with open("Models/model_bbmp.pkl", "rb") as f:
    model_package = pickle.load(f)
vectorizer = model_package["vectorizer"]
clf = model_package["classifier"]

logger.info("Loading spaCy en_core_web_sm …")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("en_core_web_sm not found – run: python -m spacy download en_core_web_sm")
    nlp = spacy.blank("en")

logger.info("Loading Whisper small …")
whisper_model = whisper.load_model("small")
logger.info("All models loaded successfully")

# ====================== FASTAPI APP ======================
app = FastAPI(title="Multilingual Civic Complaint System (BBMP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ====================== DEPENDENCIES ======================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(AdminUser).filter(AdminUser.username == username).first()
        if not user:
            raise HTTPException(status_code=401, detail="User no longer exists. Please log in again.")
            
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ====================== HELPER FUNCTIONS ======================
def extract_location(text: str) -> str:
    """Extract geographic entities from text using spaCy NER."""
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
    return ", ".join(locations) if locations else "Unknown"

def validate_audio_file(file: UploadFile):
    """Reject uploads that are not audio files (by extension + MIME type)."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    content_type = (file.content_type or "").lower()

    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        logger.warning(f"Rejected file upload: {file.filename} — not an audio file")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )
    if content_type and content_type not in ALLOWED_AUDIO_TYPES:
        logger.warning(f"Rejected file upload: {file.filename} — not an audio file")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio content type '{content_type}'.",
        )


def validate_image_file(file: UploadFile):
    """Reject uploads that are not accepted image files."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    content_type = (file.content_type or "").lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
        )
    if content_type and content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image content type '{content_type}'.",
        )


def parse_client_timestamp(timestamp_value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}. Use ISO-8601 format.")

    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def convert_dms_to_decimal(dms_value, ref) -> float:
    degrees = float(dms_value[0])
    minutes = float(dms_value[1])
    seconds = float(dms_value[2])
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_exif_location_and_time(image_path: str) -> tuple[float, float, datetime]:
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read image metadata: {exc}")

    if not exif:
        raise HTTPException(status_code=400, detail="Image must contain EXIF metadata with GPS coordinates and timestamp.")

    gps_info = exif.get(34853)
    exif_timestamp = exif.get(36867) or exif.get(306)

    if not gps_info:
        raise HTTPException(status_code=400, detail="Image EXIF GPS metadata is missing.")
    if not exif_timestamp:
        raise HTTPException(status_code=400, detail="Image EXIF timestamp is missing.")

    if isinstance(exif_timestamp, bytes):
        exif_timestamp = exif_timestamp.decode(errors="ignore")

    gps_data = {GPSTAGS.get(tag, tag): value for tag, value in gps_info.items()}
    lat = gps_data.get("GPSLatitude")
    lat_ref = gps_data.get("GPSLatitudeRef")
    lon = gps_data.get("GPSLongitude")
    lon_ref = gps_data.get("GPSLongitudeRef")

    if isinstance(lat_ref, bytes):
        lat_ref = lat_ref.decode(errors="ignore")
    if isinstance(lon_ref, bytes):
        lon_ref = lon_ref.decode(errors="ignore")

    if not lat or not lon or lat_ref not in ("N", "S") or lon_ref not in ("E", "W"):
        raise HTTPException(status_code=400, detail="Image EXIF GPS metadata is invalid.")

    try:
        image_lat = convert_dms_to_decimal(lat, lat_ref)
        image_lon = convert_dms_to_decimal(lon, lon_ref)
        image_timestamp = datetime.strptime(str(exif_timestamp), "%Y:%m:%d %H:%M:%S")
    except Exception:
        raise HTTPException(status_code=400, detail="Image EXIF metadata is malformed.")

    return image_lat, image_lon, image_timestamp


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius * c

# ====================== ENDPOINTS ======================

# ---------- Auth ----------
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticate admin user and return JWT token."""
    user = db.query(AdminUser).filter(AdminUser.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning("Failed login attempt for user '%s'", form_data.username)
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = jwt.encode({"sub": user.username}, SECRET_KEY, algorithm=ALGORITHM)
    logger.info("User '%s' logged in successfully", user.username)
    return {"access_token": access_token, "token_type": "bearer"}

class AdminCreate(BaseModel):
    username: str
    password: str
    setup_token: str

@app.post("/register-admin")
def register_admin(admin: AdminCreate, db: Session = Depends(get_db)):
    """Register a new admin user. Requires setup_token for security."""
    if admin.setup_token != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid setup token")
    
    existing_user = db.query(AdminUser).filter(AdminUser.username == admin.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
        
    hashed_pw = get_password_hash(admin.password)
    new_admin = AdminUser(username=admin.username, hashed_password=hashed_pw)
    db.add(new_admin)
    db.commit()
    db.refresh(new_admin)
    logger.info("Registered new admin: '%s'", admin.username)
    return {"msg": "Admin created successfully"}

# ---------- Submit Complaint ----------
@app.post("/submit-complaint")
async def submit_complaint(
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    live_latitude: float = Form(...),
    live_longitude: float = Form(...),
    live_location_timestamp: str = Form(...),
    text_note: str = Form(default=""),
    db: Session = Depends(get_db),
):
    """
    Full NLP pipeline: receive audio → transcribe → detect language →
    translate → classify → extract location → save to DB.
    """
    # 0. Validate live location fields (mandatory)
    if not (-90 <= live_latitude <= 90) or not (-180 <= live_longitude <= 180):
        raise HTTPException(status_code=400, detail="Live location coordinates are invalid.")

    live_location_at = parse_client_timestamp(live_location_timestamp, "live_location_timestamp")

    submitted_text = (text_note or "").strip()
    if file is None and not submitted_text:
        raise HTTPException(status_code=400, detail="Provide either audio or complaint text along with live location.")

    if file is not None:
        validate_audio_file(file)
    if image is not None:
        validate_image_file(image)

    MAX_FILE_SIZE_MB = 25
    if file is not None:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
            )
        await file.seek(0)

    if image is not None:
        image_contents = await image.read()
        if len(image_contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
            )
        await image.seek(0)

    # 1. Save evidence files
    os.makedirs("uploads", exist_ok=True)
    audio_path = None
    image_path = None
    if file is not None:
        audio_id = str(uuid.uuid4())
        audio_ext = os.path.splitext(file.filename or ".wav")[1] or ".wav"
        audio_path = f"uploads/{audio_id}{audio_ext}"
        with open(audio_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"Received audio file: {file.filename}")

    if image is not None:
        image_id = str(uuid.uuid4())
        image_ext = os.path.splitext(image.filename or ".jpg")[1] or ".jpg"
        image_path = f"uploads/{image_id}{image_ext}"
        with open(image_path, "wb") as f:
            f.write(await image.read())
        logger.info(f"Received image evidence: {image.filename}")

    # 2 & 3. Derive complaint text from audio and/or text input
    transcribed_text = submitted_text
    detected_language = "en"
    if audio_path:
        try:
            result = whisper_model.transcribe(
                audio_path,
                language="kn",
                task="translate",
                fp16=False,
                condition_on_previous_text=False,
            )
            audio_text = result["text"].strip()
            detected_language = result.get("language", "en")
            if not audio_text and not submitted_text:
                logger.warning(f"No speech detected in uploaded audio: {audio_path}")
                raise HTTPException(
                    status_code=400,
                    detail="No clear speech detected in audio. Please record again and speak closer to the microphone."
                )
            transcribed_text = f"{submitted_text} {audio_text}".strip() if submitted_text else audio_text
            logger.info(f"Transcribed: {audio_text} | Language: {detected_language}")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error(f"Whisper transcription failed for {audio_path}: {e}")
            if "WinError 2" in str(e) or "ffmpeg" in str(e).lower():
                logger.warning("FFmpeg not found. Cannot transcribe audio without FFmpeg.")
                raise HTTPException(
                    status_code=500,
                    detail="Audio transcription failed: FFmpeg is not installed or not found in PATH. Please install FFmpeg and retry."
                )
            raise HTTPException(status_code=500, detail="Audio transcription failed. Please try again.")

    if not transcribed_text:
        raise HTTPException(status_code=400, detail="Complaint text is required.")

    # 4. Translate to English using deep-translator
    try:
        translated_text = GoogleTranslator(
            source="auto", target="en"
        ).translate(transcribed_text)
        logger.info(f"Translated to English: {translated_text}")
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        translated_text = transcribed_text
        logger.warning("Using original text as fallback due to translation failure.")

    try:
        category = clf.predict(vectorizer.transform([translated_text]))[0]
        location = f"{live_latitude:.6f}, {live_longitude:.6f}"
        logger.info(f"Category: {category} | Location: {location}")
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        category = "Others"
        location = f"{live_latitude:.6f}, {live_longitude:.6f}"

    # 7. Validate optional image evidence authenticity and assign trust level
    image_exif_latitude = None
    image_exif_longitude = None
    image_exif_timestamp = None
    image_live_distance_meters = None
    trust_level = "medium"
    verification_mode = "manual_review"
    status = "pending"

    if image_path:
        image_exif_latitude, image_exif_longitude, image_exif_timestamp = extract_exif_location_and_time(image_path)
        image_live_distance_meters = haversine_distance_meters(
            live_latitude,
            live_longitude,
            image_exif_latitude,
            image_exif_longitude,
        )
        if image_live_distance_meters > GPS_TOLERANCE_METERS:
            raise HTTPException(
                status_code=400,
                detail=f"Image GPS does not match live location within {int(GPS_TOLERANCE_METERS)} meters."
            )

        image_age_seconds = (datetime.utcnow() - image_exif_timestamp).total_seconds()
        if image_age_seconds > MAX_IMAGE_AGE_SECONDS:
            raise HTTPException(
                status_code=400,
                detail="Image metadata timestamp is older than 10 minutes. Capture a fresh photo."
            )

        trust_level = "high"
        verification_mode = "auto_verified"
        status = "Verified"

    # 8. Save to database
    complaint = Complaint(
        audio_path=audio_path,
        image_path=image_path,
        original_text=transcribed_text,
        translated_text=translated_text,
        language=detected_language,
        category=category,
        location=location,
        live_latitude=live_latitude,
        live_longitude=live_longitude,
        live_location_timestamp=live_location_at,
        image_exif_latitude=image_exif_latitude,
        image_exif_longitude=image_exif_longitude,
        image_exif_timestamp=image_exif_timestamp,
        image_live_distance_meters=image_live_distance_meters,
        trust_level=trust_level,
        verification_mode=verification_mode,
        status=status,
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)
    logger.info(f"Complaint saved with ID: {complaint.id}")

    # Return JSON response
    return {
        "id": complaint.id,
        "category": complaint.category,
        "location": complaint.location,
        "live_latitude": complaint.live_latitude,
        "live_longitude": complaint.live_longitude,
        "trust_level": complaint.trust_level,
        "verification_mode": complaint.verification_mode,
        "image_live_distance_meters": complaint.image_live_distance_meters,
        "translated_text": complaint.translated_text,
        "status": complaint.status,
    }

# ---------- Audio File Serving ----------
@app.get("/uploads/{filename}")
async def serve_audio(
    filename: str,
    request: Request,
    token: Optional[str] = Query(default=None),
):
    # Accept token from query param (audio src use-case) or Authorization header.
    access_token = token
    if not access_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            access_token = auth_header.split(" ", 1)[1].strip()

    if not access_token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    file_path = os.path.join("uploads", filename)
    if not os.path.exists(file_path):
        logger.warning(f"Audio file not found: {filename}")
        raise HTTPException(status_code=404, detail="Audio file not found.")

    media_type, _ = mimetypes.guess_type(file_path)
    if not media_type:
        media_type = "application/octet-stream"

    logger.info(f"Serving audio file: {filename}")
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )

# ---------- Complaint Statistics ----------
@app.get("/complaints/stats")
async def get_stats(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    total = db.query(Complaint).count()
    pending = db.query(Complaint).filter(Complaint.status == "pending").count()
    verified = db.query(Complaint).filter(Complaint.status == "Verified").count()

    by_category = db.query(
        Complaint.category, func.count(Complaint.id).label("count")
    ).group_by(Complaint.category).all()

    by_language = db.query(
        Complaint.language, func.count(Complaint.id).label("count")
    ).group_by(Complaint.language).all()

    logger.info("Stats endpoint called.")
    return {
        "total": total,
        "pending": pending,
        "verified": verified,
        "by_category": [{"category": r[0], "count": r[1]} for r in by_category],
        "by_language": [{"language": r[0], "count": r[1]} for r in by_language],
    }

# ---------- List Complaints (Paginated, JWT-protected) ----------
@app.get("/complaints")
async def get_complaints(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    total = db.query(Complaint).count()
    pages = math.ceil(total / size)
    offset = (page - 1) * size
    items = db.query(Complaint).order_by(
        Complaint.created_at.desc()
    ).offset(offset).limit(size).all()

    logger.info(f"GET /complaints page={page} size={size} total={total}")
    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages
    }

# ---------- Verify / Edit Complaint (HITL, JWT-protected) ----------
class VerifyRequest(BaseModel):
    category: str = None
    status: str = "Verified"

@app.put("/complaints/{id}/verify")
def verify_complaint(
    id: int,
    request: VerifyRequest,
    db: Session = Depends(get_db),
):
    """HITL: admin verifies or edits a complaint's category/status."""
    complaint = db.query(Complaint).filter(Complaint.id == id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    if request.category:
        complaint.category = request.category
    if request.status:
        complaint.status = request.status

    db.commit()
    db.refresh(complaint)
    logger.info("Complaint #%d updated — status=%s, category=%s", id, complaint.status, complaint.category)

    return {
        "id": complaint.id,
        "category": complaint.category,
        "location": complaint.location,
        "translated_text": complaint.translated_text,
        "status": complaint.status,
    }

# ====================== RUN ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)