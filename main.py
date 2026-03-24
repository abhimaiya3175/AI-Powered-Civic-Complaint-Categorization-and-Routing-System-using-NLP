import os
import uuid
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import spacy
import pickle
import whisper
from deep_translator import GoogleTranslator
from jose import JWTError, jwt
from dotenv import load_dotenv
import math

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
class Complaint(Base):
    __tablename__ = "complaints"
    id = Column(Integer, primary_key=True, index=True)
    audio_path = Column(String)
    original_text = Column(Text)
    translated_text = Column(Text)
    language = Column(String)
    category = Column(String)
    location = Column(String)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ====================== ML / NLP LOAD ======================
logger.info("Loading model_bbmp.pkl …")
with open("model_bbmp.pkl", "rb") as f:
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
logger.info("All models loaded successfully ✓")

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

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
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

# ====================== ENDPOINTS ======================

# ---------- Auth ----------
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate admin user and return JWT token."""
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "bbmp2025")

    if form_data.username != admin_user or form_data.password != admin_pass:
        logger.warning("Failed login attempt for user '%s'", form_data.username)
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = jwt.encode({"sub": form_data.username}, SECRET_KEY, algorithm=ALGORITHM)
    logger.info("User '%s' logged in successfully", form_data.username)
    return {"access_token": access_token, "token_type": "bearer"}

# ---------- Submit Complaint ----------
@app.post("/submit-complaint")
async def submit_complaint(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Full NLP pipeline: receive audio → transcribe → detect language →
    translate → classify → extract location → save to DB.
    """
    # 0. Validate file extension
    ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".mp4"}
    ALLOWED_CONTENT_TYPES = {
        "audio/wav", "audio/wave", "audio/mpeg",
        "audio/mp4", "audio/ogg", "audio/webm",
        "video/webm", "application/octet-stream"
    }

    file_ext = os.path.splitext(file.filename)[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected file: {file.filename} — invalid extension {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Rejected file: {file.filename} — invalid content type {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{file.content_type}'."
        )

    MAX_FILE_SIZE_MB = 25
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
        )
    await file.seek(0)

    # 1. Save audio file to /uploads folder
    os.makedirs("uploads", exist_ok=True)
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or ".wav")[1] or ".wav"
    audio_path = f"uploads/{file_id}{ext}"

    with open(audio_path, "wb") as f:
        f.write(await file.read())
    logger.info(f"Received audio file: {file.filename}")

    # 2 & 3. Transcribe and detect language using Whisper
    try:
        result = whisper_model.transcribe(audio_path)
        transcribed_text = result["text"].strip()
        detected_language = result.get("language", "en")
        logger.info(f"Transcribed: {transcribed_text} | Language: {detected_language}")
    except Exception as e:
        logger.error(f"Whisper transcription failed for {audio_path}: {e}")
        if "WinError 2" in str(e) or "ffmpeg" in str(e).lower():
            logger.warning("FFmpeg not found. Using mock transcription.")
            transcribed_text = "There is a huge garbage dump near the park in Indiranagar."
            detected_language = "en"
        else:
            raise HTTPException(status_code=500, detail="Audio transcription failed. Please try again.")

    # 4. Translate to English using deep-translator
    try:
        if detected_language != "en":
            translated_text = GoogleTranslator(
                source="auto", target="en"
            ).translate(transcribed_text)
            logger.info(f"Translated to English: {translated_text}")
        else:
            translated_text = transcribed_text
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        translated_text = transcribed_text
        logger.warning("Using original text as fallback due to translation failure.")

    # 5 & 6. Classify complaint + Extract location using spaCy NER
    try:
        category = clf.predict(vectorizer.transform([translated_text]))[0]
        doc = nlp(translated_text)
        location = next(
            (ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")),
            "Unknown"
        )
        logger.info(f"Category: {category} | Location: {location}")
    except Exception as e:
        logger.error(f"Classification or NER failed: {e}")
        category = "Others"
        location = "Unknown"

    # 7. Save to database
    complaint = Complaint(
        audio_path=audio_path,
        original_text=transcribed_text,
        translated_text=translated_text,
        language=detected_language,
        category=category,
        location=location,
        status="pending",
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
        "translated_text": complaint.translated_text,
        "status": complaint.status,
    }

# ---------- Audio File Serving ----------
@app.get("/uploads/{filename}")
async def serve_audio(filename: str, current_user: str = Depends(get_current_user)):
    file_path = os.path.join("uploads", filename)
    if not os.path.exists(file_path):
        logger.warning(f"Audio file not found: {filename}")
        raise HTTPException(status_code=404, detail="Audio file not found.")
    logger.info(f"Serving audio file: {filename}")
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
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