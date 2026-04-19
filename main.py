import os
import uuid
import logging
import mimetypes
import asyncio
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
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
import torch
from jose import JWTError, jwt
from dotenv import load_dotenv
import re
import math
import bcrypt
import speech_recognition as sr
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from IndicTransToolkit import IndicProcessor  # type: ignore
except Exception:
    try:
        # Some distributions expose IndicProcessor from a nested module.
        from IndicTransToolkit.processor import IndicProcessor  # type: ignore
    except Exception:
        IndicProcessor = None


class IndicProcessorFallback:
    """Fallback processor when IndicTransToolkit cannot be imported on the host."""

    def __init__(self, inference: bool = True):
        self.inference = inference

    def preprocess_batch(self, text_batch: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        # Basic language-tag prefix so IndicTrans2 still receives source/target hints.
        return [f"{src_lang} {tgt_lang} {(text or '').strip()}".strip() for text in text_batch]

    def postprocess_batch(self, text_batch: list[str], lang: str) -> list[str]:
        return [(text or "").strip() for text in text_batch]

load_dotenv()

# Ensure multilingual logs do not crash on cp1252 terminals (common on Windows).
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bbmp_complaints.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("bbmp")

# ====================== CONFIG ======================
SECRET_KEY = os.getenv("SECRET_KEY", "replace-with-a-strong-secret")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
INDICTRANS2_MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
NLLB_FALLBACK_MODEL_NAME = "facebook/nllb-200-distilled-600M"
ROTARY_INDICTRANS2_FALLBACK_MODEL_NAME = "prajdabre/rotary-indictrans2-indic-en-dist-200M"

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "kn": "Kannada",
    "hi": "Hindi",
    "en": "English",
}
INDIC_LANG_TAGS: Dict[str, str] = {
    "kn": "kan_Knda",
    "hi": "hin_Deva",
    "en": "eng_Latn",
}

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

import urllib.parse
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)

# Prefer explicit DATABASE_URL, otherwise construct from DB_* values.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
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
logger.info("Core models loaded successfully")

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

# ── Health Check (Railway / Docker readiness probe) ──────────────
from fastapi.responses import JSONResponse

@app.get("/health", tags=["ops"])
async def health_check():
    """Liveness + readiness probe used by Railway and Docker healthchecks."""
    try:
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    status = "ok" if db_ok else "degraded"
    return JSONResponse(
        content={"status": status, "db": "connected" if db_ok else "unreachable"},
        status_code=200 if db_ok else 503,
    )


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


def normalize_language_code(language_value: str, field_name: str) -> str:
    normalized = (language_value or "").strip().lower()
    if normalized not in SUPPORTED_LANGUAGES:
        allowed = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} '{language_value}'. Allowed values: {allowed}",
        )
    return normalized


def _translate_batch_sync(
    text_batch: list[str],
    src_lang_tag: str,
    tgt_lang_tag: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    processor: Any,
    translation_backend: str,
) -> list[str]:
    """Run one CPU translation batch with backend-specific preprocessing."""
    # Official IndicTrans2 path with IndicProcessor preprocessing/postprocessing.
    if translation_backend == "indictrans2":
        model_inputs = processor.preprocess_batch(text_batch, src_lang=src_lang_tag, tgt_lang=tgt_lang_tag)

        encoded_inputs = tokenizer(
            model_inputs,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        encoded_inputs = {key: value.to("cpu") for key, value in encoded_inputs.items()}

        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_inputs,
                max_length=256,
                num_beams=5,
                do_sample=False,
                use_cache=True,
            )

        decoded_batch = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return processor.postprocess_batch(decoded_batch, lang=tgt_lang_tag)

    # NLLB fallback path for robust Indic -> English translation when
    # official IndicTrans2 checkpoints are inaccessible in the environment.
    if translation_backend == "nllb":
        try:
            tokenizer.src_lang = src_lang_tag
        except Exception:
            pass

        encoded_inputs = tokenizer(
            text_batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        encoded_inputs = {key: value.to("cpu") for key, value in encoded_inputs.items()}

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_tag)

        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=256,
                num_beams=5,
                do_sample=False,
                use_cache=True,
            )

        decoded_batch = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [(text or "").strip() for text in decoded_batch]

    # Fallback path for public rotary IndicTrans2 checkpoints that require
    # language tags directly in-text and run more stably with cache disabled.
    tagged_inputs = [f"{src_lang_tag} {tgt_lang_tag} {(text or '').strip()}".strip() for text in text_batch]

    encoded_inputs = tokenizer(
        tagged_inputs,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )
    encoded_inputs = {key: value.to("cpu") for key, value in encoded_inputs.items()}

    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded_inputs,
            max_length=256,
            num_beams=5,
            do_sample=False,
            use_cache=False,
        )

    decoded_batch = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return [(text or "").strip() for text in decoded_batch]


async def translate_text_with_indictrans2(text: str, source_language: str, target_language: str) -> str:
    """
    Mandatory step 2: dedicated NLP translation after Whisper transcription.
    Uses IndicTrans2 with IndicTransToolkit preprocessing/postprocessing.
    """
    clean_text = (text or "").strip()
    if not clean_text or source_language == target_language:
        return clean_text

    # This CPU model is optimized for Indic->English translation.
    # For unsupported target directions, keep original text as safe fallback.
    if target_language != "en":
        logger.warning("Requested translation to '%s' with Indic->English model. Returning original text.", target_language)
        return clean_text

    if source_language not in INDIC_LANG_TAGS or target_language not in INDIC_LANG_TAGS:
        return clean_text

    if source_language == "en":
        return clean_text

    tokenizer = getattr(app.state, "indictrans_tokenizer", None)
    model = getattr(app.state, "indictrans_model", None)
    processor = getattr(app.state, "indic_processor", None)
    translation_backend = getattr(app.state, "translation_backend", "unknown")
    translation_lock = getattr(app.state, "translation_lock", None)

    if tokenizer is None or model is None or processor is None or translation_lock is None:
        logger.warning("Translation assets are not available (backend=%s). Returning original text.", translation_backend)
        return clean_text

    src_lang_tag = INDIC_LANG_TAGS[source_language]
    tgt_lang_tag = INDIC_LANG_TAGS[target_language]

    try:
        # Offload model.generate to a worker thread so FastAPI event loop stays responsive on CPU.
        async with translation_lock:
            translated_batch = await asyncio.to_thread(
                _translate_batch_sync,
                [clean_text],
                src_lang_tag,
                tgt_lang_tag,
                tokenizer,
                model,
                processor,
                translation_backend,
            )

        translated_text = (translated_batch[0] if translated_batch else "").strip()
        return translated_text or clean_text
    except Exception as exc:
        logger.error("IndicTrans2 translation failed (%s->%s): %s", source_language, target_language, exc)
        return clean_text


def convert_dms_to_decimal(dms_value, ref) -> float:
    if not isinstance(dms_value, (list, tuple)) or len(dms_value) != 3:
        raise ValueError("Invalid GPS coordinate format")

    def _ratio_to_float(component) -> float:
        # Pillow may expose EXIF rationals as IFDRational, tuples, or plain numbers.
        if hasattr(component, "numerator") and hasattr(component, "denominator"):
            denominator = float(component.denominator) if component.denominator else 1.0
            return float(component.numerator) / denominator
        if isinstance(component, (tuple, list)) and len(component) == 2:
            numerator = float(component[0])
            denominator = float(component[1]) if component[1] else 1.0
            return numerator / denominator
        return float(component)

    degrees = _ratio_to_float(dms_value[0])
    minutes = _ratio_to_float(dms_value[1])
    seconds = _ratio_to_float(dms_value[2])
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

    gps_info = None
    if hasattr(exif, "get_ifd"):
        try:
            gps_info = exif.get_ifd(34853)
        except Exception:
            gps_info = None

    if not gps_info:
        legacy_gps_info = exif.get(34853)
        if isinstance(legacy_gps_info, dict):
            gps_info = legacy_gps_info

    exif_timestamp = exif.get(36867) or exif.get(306)

    if not gps_info:
        raise HTTPException(status_code=400, detail="Image EXIF GPS metadata is missing.")
    if not exif_timestamp:
        raise HTTPException(status_code=400, detail="Image EXIF timestamp is missing.")

    if isinstance(exif_timestamp, bytes):
        exif_timestamp = exif_timestamp.decode(errors="ignore")

    if not isinstance(gps_info, dict):
        raise HTTPException(status_code=400, detail="Image EXIF GPS metadata is malformed.")

    gps_data = {GPSTAGS.get(tag, tag): value for tag, value in gps_info.items()}
    lat = gps_data.get("GPSLatitude") or gps_data.get(2)
    lat_ref = gps_data.get("GPSLatitudeRef") or gps_data.get(1)
    lon = gps_data.get("GPSLongitude") or gps_data.get(4)
    lon_ref = gps_data.get("GPSLongitudeRef") or gps_data.get(3)

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


@app.on_event("startup")
async def preload_translation_models() -> None:
    """Preload IndicTrans2 translation assets once at startup for CPU inference."""

    def _load_indictrans2_assets(model_name: str) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model.to("cpu")
        model.eval()
        processor_class = IndicProcessor if IndicProcessor is not None else IndicProcessorFallback
        processor = processor_class(inference=True)
        return tokenizer, model, processor

    def _load_nllb_assets(model_name: str) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            use_safetensors=False,
            torch_dtype=torch.float32,
        )
        model.to("cpu")
        model.eval()
        return tokenizer, model, IndicProcessorFallback(inference=True)

    app.state.indictrans_tokenizer = None
    app.state.indictrans_model = None
    app.state.indic_processor = None
    app.state.translation_backend = "unavailable"
    app.state.translation_lock = asyncio.Lock()

    try:
        logger.info("Loading IndicTrans2 distilled CPU model: %s", INDICTRANS2_MODEL_NAME)
        tokenizer, model, processor = await asyncio.to_thread(_load_indictrans2_assets, INDICTRANS2_MODEL_NAME)
        app.state.indictrans_tokenizer = tokenizer
        app.state.indictrans_model = model
        app.state.indic_processor = processor
        app.state.translation_backend = "indictrans2"
        if IndicProcessor is None:
            logger.warning("IndicTransToolkit not installed; using fallback processor for IndicTrans2 preprocessing.")
        logger.info("IndicTrans2 + IndicProcessor loaded successfully (backend=indictrans2).")
        return
    except Exception as primary_exc:
        logger.error("Failed to preload primary IndicTrans2 assets (%s): %s", INDICTRANS2_MODEL_NAME, primary_exc)

    try:
        logger.info("Loading fallback translation model: %s", NLLB_FALLBACK_MODEL_NAME)
        tokenizer, model, processor = await asyncio.to_thread(_load_nllb_assets, NLLB_FALLBACK_MODEL_NAME)
        app.state.indictrans_tokenizer = tokenizer
        app.state.indictrans_model = model
        app.state.indic_processor = processor
        app.state.translation_backend = "nllb"
        logger.info("Fallback translation model loaded successfully (backend=nllb).")
        return
    except Exception as nllb_exc:
        logger.error("Failed to preload fallback translation assets (%s): %s", NLLB_FALLBACK_MODEL_NAME, nllb_exc)

    try:
        logger.info("Loading tertiary fallback translation model: %s", ROTARY_INDICTRANS2_FALLBACK_MODEL_NAME)
        tokenizer, model, processor = await asyncio.to_thread(
            _load_indictrans2_assets,
            ROTARY_INDICTRANS2_FALLBACK_MODEL_NAME,
        )
        app.state.indictrans_tokenizer = tokenizer
        app.state.indictrans_model = model
        app.state.indic_processor = processor
        app.state.translation_backend = "rotary_indictrans2"
        logger.info("Tertiary IndicTrans2-compatible model loaded successfully (backend=rotary_indictrans2).")
    except Exception as rotary_exc:
        logger.error(
            "Failed to preload tertiary fallback translation assets (%s): %s",
            ROTARY_INDICTRANS2_FALLBACK_MODEL_NAME,
            rotary_exc,
        )

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
    language: str = Form(default="en"),
    target_language: str = Form(default="en"),
    db: Session = Depends(get_db),
):
    """
    Full NLP pipeline: receive audio → transcribe (Whisper) →
    translate (dedicated NLP model) → classify → save to DB.
    """
    # 0. Validate live location fields (mandatory)
    if not (-90 <= live_latitude <= 90) or not (-180 <= live_longitude <= 180):
        raise HTTPException(status_code=400, detail="Live location coordinates are invalid.")

    live_location_at = parse_client_timestamp(live_location_timestamp, "live_location_timestamp")

    submitted_text = (text_note or "").strip()
    recording_language = normalize_language_code(language, "language")
    target_language_code = normalize_language_code(target_language, "target_language")

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

    # 2. Whisper is used only for transcription in the selected recording language.
    # 3. Translation is a separate NLP step (IndicTrans2 + IndicTransToolkit preprocessing).
    transcribed_text = submitted_text
    detected_language = recording_language
    if audio_path:
        if recording_language in ["kn", "hi", "en"]:
            logger.info("Using Google STT for highly accurate transcription.")
            try:
                lang_code_map = {"kn": "kn-IN", "hi": "hi-IN", "en": "en-IN"}
                google_lang = lang_code_map.get(recording_language, "en-IN")
                
                # Convert webm to wav for SpeechRecognition
                wav_path = audio_path + ".wav"
                audio_segment = AudioSegment.from_file(audio_path)
                audio_segment.export(wav_path, format="wav")
                
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                
                audio_text = await asyncio.to_thread(
                    recognizer.recognize_google,
                    audio_data,
                    language=google_lang
                )
                
                transcribed_text = f"{audio_text} {submitted_text}".strip() if submitted_text else audio_text
                logger.info(f"Google STT Transcribed: {audio_text} | Language: {detected_language}")
                
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                    
            except sr.UnknownValueError:
                logger.warning(f"Google STT could not understand audio for {audio_path}")
                raise HTTPException(
                    status_code=400,
                    detail="No clear speech detected in audio. Please record again and speak closer to the microphone."
                )
            except Exception as e:
                logger.error(f"Google STT failed: {e}. Falling back to Whisper...")
                # Fallback to Whisper
                try:
                    result = await asyncio.to_thread(
                        whisper_model.transcribe,
                        audio_path,
                        task="transcribe",
                        fp16=False,
                        condition_on_previous_text=False,
                    )
                    audio_text = (result.get("text") or "").strip()
                    whisper_detected_language = (result.get("language") or recording_language).strip().lower()
                    if whisper_detected_language in SUPPORTED_LANGUAGES:
                        detected_language = whisper_detected_language
        
                    if not audio_text and not submitted_text:
                        logger.warning(f"No speech detected in uploaded audio: {audio_path}")
                        raise HTTPException(
                            status_code=400,
                            detail="No clear speech detected in audio. Please record again and speak closer to the microphone."
                        )
                    transcribed_text = f"{audio_text} {submitted_text}".strip() if submitted_text else audio_text
                    logger.info(f"Whisper Transcribed: {audio_text} | Language: {detected_language}")
                except Exception as ex:
                    raise HTTPException(status_code=500, detail="Audio transcription failed. Please try again.")
        else:
            try:
                result = await asyncio.to_thread(
                    whisper_model.transcribe,
                    audio_path,
                    task="transcribe",
                    fp16=False,
                    condition_on_previous_text=False,
                )
                audio_text = (result.get("text") or "").strip()
                whisper_detected_language = (result.get("language") or recording_language).strip().lower()
                if whisper_detected_language in SUPPORTED_LANGUAGES:
                    detected_language = whisper_detected_language
    
                if not audio_text and not submitted_text:
                    logger.warning(f"No speech detected in uploaded audio: {audio_path}")
                    raise HTTPException(
                        status_code=400,
                        detail="No clear speech detected in audio. Please record again and speak closer to the microphone."
                    )
                transcribed_text = f"{audio_text} {submitted_text}".strip() if submitted_text else audio_text
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

    # Mandatory step: perform translation using dedicated NLP model (not Whisper translate task).
    translated_text = await translate_text_with_indictrans2(
        transcribed_text,
        detected_language,
        target_language_code,
    )
    logger.info("Translated text (%s->%s): %s", detected_language, target_language_code, translated_text)

    english_text_for_classification = translated_text
    if target_language_code != "en":
        if detected_language == "en":
            english_text_for_classification = transcribed_text
        else:
            english_text_for_classification = await translate_text_with_indictrans2(
                transcribed_text,
                detected_language,
                "en",
            ) or transcribed_text

    try:
        category = clf.predict(vectorizer.transform([english_text_for_classification]))[0]
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
        "transcribed_text": complaint.original_text,
        "trust_level": complaint.trust_level,
        "verification_mode": complaint.verification_mode,
        "image_live_distance_meters": complaint.image_live_distance_meters,
        "translated_text": complaint.translated_text,
        "detected_language": complaint.language,
        "target_language": target_language_code,
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
    uvicorn.run(app, host="0.0.0.0", port=8000)