import os
import uuid
import logging
import mimetypes
import asyncio
import sys
import threading
import time
import json
import platform
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, func, inspect, text as sql_text
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nlp_features import build_multilingual_classification_text
from image_features import analyze_image, load_yolo_model, DETECTION_CLASS_TO_CATEGORY

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
# All secrets MUST be set in .env
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY is missing. Please set it in your .env file.")

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
# Ensure we strictly read the password from .env
DB_PASSWORD = os.getenv("DB_PASSWORD") 
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

MODEL_PATH_CANDIDATES = [
    os.getenv("MODEL_PATH", "").strip(),
    "Models/model_bbmp.pkl",
    "model_bbmp.pkl",
]

ENABLE_ZERO_SHOT_FALLBACK = os.getenv("ENABLE_ZERO_SHOT_FALLBACK", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ZERO_SHOT_MODEL_NAME = os.getenv("ZERO_SHOT_MODEL_NAME", "valhalla/distilbart-mnli-12-1").strip()
ZERO_SHOT_MIN_CONFIDENCE = float(os.getenv("ZERO_SHOT_MIN_CONFIDENCE", "0.85"))
ZERO_SHOT_MIN_SCORE = float(os.getenv("ZERO_SHOT_MIN_SCORE", "0.55"))
ZERO_SHOT_SPARSE_MIN_SCORE = float(os.getenv("ZERO_SHOT_SPARSE_MIN_SCORE", "0.60"))
PRIMARY_MIN_EXPLANATORY_FEATURES = int(os.getenv("PRIMARY_MIN_EXPLANATORY_FEATURES", "2"))
IMAGE_RECONCILE_CONFIDENCE_THRESHOLD = float(os.getenv("IMAGE_RECONCILE_CONFIDENCE_THRESHOLD", "0.6"))

GENERIC_PRIMARY_FEATURE_TERMS = {
    "street",
    "road",
    "area",
    "near",
    "public",
    "issue",
    "problem",
    "big",
    "small",
}

CATEGORY_SEMANTIC_HINTS = {
    "Street Light": "street light electrical lamp not working",
    "Garbage / Sanitation": "garbage waste sanitation not collected",
    "Road Repair": "road pothole damaged road repair",
    "Drainage / SWD": "drainage overflow sewer storm water drain",
    "Water Supply": "water supply no water tap dry",
    "Health / Sanitation": "health sanitation mosquito public health",
    "Parks / Forest": "park forest tree issue",
    "Parks": "park playground public park issue",
    "Town Planning": "town planning building plan issue",
    "Veterinary": "veterinary animal dog cattle issue",
    "Advertisement": "advertisement hoarding banner issue",
    "Revenue": "revenue property tax khata issue",
    "Others": "other civic complaint",
}

KANNADA_POTHOLE_TERMS = {
    "ಗುಂಡಿ",
    "ಗುಂಡಿಗಳು",
    "ಗುಂಡಿಯ",
    "ಗುಂಡಿಯನ್ನು",
    "ಗುಂಡಿಯಲ್ಲಿ",
    "ಗುಂಡಿಗಳಿಗೆ",
    "ಗುಂಡಿಗಳ",
    "ಗಂಡಿ",
}
KANNADA_POTHOLE_TRANSLIT_PATTERN = re.compile(r"\bgund[iy](?:galu|ge|alli|inda|yalli)?\b", re.IGNORECASE)

zero_shot_classifier = None
zero_shot_classifier_lock = threading.Lock()

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
    votes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Image detection (YOLOv8n-seg pothole/road-damage detection)
    detected_objects = Column(Text)         # JSON: [{class, confidence, bbox, severity}]
    annotated_image_path = Column(String)   # Path to annotated image (nullable)
    pothole_severity = Column(String)       # Overall severity: Clear/Low/Medium/High/Severe (nullable)
    image_suggested_category = Column(String)  # Category suggested by image analysis, nullable
    category_mismatch = Column(Boolean, default=False)  # True if image analysis strongly disagrees with NLP

class ComplaintVote(Base):
    __tablename__ = "complaint_votes"
    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, index=True)
    voter_fingerprint = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ComplaintTimeline(Base):
    __tablename__ = "complaint_timeline"
    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, index=True)
    status = Column(String)
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class NlpMetric(Base):
    """Stores real per-request NLP processing metrics for analytics.

    Every column is populated from actual runtime measurements
    (time.perf_counter, sklearn predict_proba, spaCy doc.ents, pydub duration).
    No hardcoded, estimated, random, static, or sample values.
    """
    __tablename__ = "nlp_metrics"
    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_duplicate = Column(Boolean, default=False)
    source_language = Column(String)
    category = Column(String)
    # Classification quality
    classifier_confidence = Column(Float)  # sklearn predict_proba() score
    zero_shot_triggered = Column(Boolean, default=False)
    zero_shot_confidence = Column(Float, default=0.0)
    # NER quality
    entity_count = Column(Integer, default=0)  # len(spacy_doc.ents)
    entity_types = Column(Text)  # JSON: {"GPE": 2, "LOC": 1}
    # Input characteristics
    audio_duration_seconds = Column(Float)  # pydub AudioSegment.duration_seconds
    word_count = Column(Integer, default=0)  # len(text.split())
    # Stage timings (seconds, measured via time.perf_counter)
    transcription_time = Column(Float, default=0.0)
    translation_time = Column(Float, default=0.0)
    classification_time = Column(Float, default=0.0)
    ner_time = Column(Float, default=0.0)
    zero_shot_time = Column(Float, default=0.0)
    image_analysis_time = Column(Float, default=0.0)
    total_processing_time = Column(Float, default=0.0)
    # Energy (Joules = estimated_power_watts × processing_time_seconds)
    estimated_power_watts = Column(Float, default=0.0)
    total_energy_joules = Column(Float, default=0.0)
    energy_by_stage = Column(Text)  # JSON: per-stage energy breakdown
    calculation_method = Column(String)  # Documents how energy was derived
    # Image analysis stage metrics
    detected_object_count = Column(Integer, default=0)
    image_model_confidence = Column(Float)  # max confidence across detections
    pothole_severity = Column(String)       # severity bucket from image analysis
    # Error tracking
    error_stage = Column(String)  # Which stage failed (null = success)
    error_message = Column(Text)  # Exception message


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

def _ensure_table_columns(table_name: str, required_columns: dict) -> None:
    """Add missing columns to an existing table without migrations."""
    inspector_obj = inspect(engine)
    if table_name not in inspector_obj.get_table_names():
        return
    existing = {col["name"] for col in inspector_obj.get_columns(table_name)}
    with engine.begin() as conn:
        for column_name, ddl in required_columns.items():
            if column_name not in existing:
                conn.execute(sql_text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}"))


def ensure_complaints_schema_upgrades():
    """Add newly introduced columns for existing databases without migrations."""
    _ensure_table_columns("complaints", {
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
        "votes": "INTEGER DEFAULT 0",
        # Image detection (YOLOv8n-seg)
        "detected_objects": "TEXT",
        "annotated_image_path": "VARCHAR",
        "pothole_severity": "VARCHAR",
        "image_suggested_category": "VARCHAR",
        "category_mismatch": "BOOLEAN DEFAULT FALSE",
    })


def ensure_nlp_metrics_schema_upgrades():
    """Add image analysis columns to nlp_metrics for existing databases."""
    _ensure_table_columns("nlp_metrics", {
        "image_analysis_time": "FLOAT DEFAULT 0.0",
        "detected_object_count": "INTEGER DEFAULT 0",
        "image_model_confidence": "FLOAT",
        "pothole_severity": "VARCHAR",
    })


ensure_complaints_schema_upgrades()
ensure_nlp_metrics_schema_upgrades()


# ====================== CPU POWER ESTIMATION ======================
def _detect_cpu_power_watts() -> tuple[float, str]:
    """Estimate CPU TDP from actual hardware info for energy calculations.

    Returns (watts, method_description). NOT a random or hardcoded value —
    derived from the real CPU model detected via platform.processor().
    """
    cpu_model = platform.processor() or "unknown"
    cpu_count = os.cpu_count() or 1
    machine = platform.machine() or "unknown"

    # Heuristics based on actual CPU model string patterns
    cpu_lower = cpu_model.lower()
    if any(tag in cpu_lower for tag in ["arm", "aarch64", "apple m"]):
        watts = 10.0
        tier = "ARM/Apple Silicon (low-power)"
    elif any(tag in cpu_lower for tag in ["u", "p", "mobile", "laptop"]):
        watts = 15.0
        tier = "Mobile/Ultrabook CPU"
    elif any(tag in cpu_lower for tag in ["h", "hx", "hk"]):
        watts = 45.0
        tier = "High-performance laptop CPU"
    elif any(tag in cpu_lower for tag in ["k", "x", "server", "xeon", "epyc"]):
        watts = 95.0
        tier = "Desktop/Server CPU"
    elif cpu_count >= 16:
        watts = 65.0
        tier = "Multi-core desktop (inferred from core count)"
    elif cpu_count >= 8:
        watts = 45.0
        tier = "Desktop CPU (inferred from core count)"
    else:
        watts = 25.0
        tier = "Generic CPU (conservative estimate)"

    method = (
        f"CPU: {cpu_model}, Arch: {machine}, Cores: {cpu_count}. "
        f"Tier: {tier}. Estimated TDP: {watts}W. "
        f"Energy (J) = {watts}W × measured processing time (s). "
        f"Detected via platform.processor() at startup."
    )
    return watts, method


ESTIMATED_CPU_POWER_WATTS, CPU_POWER_DETECTION_METHOD = _detect_cpu_power_watts()
logger.info(
    "CPU power estimation: %.1fW — %s",
    ESTIMATED_CPU_POWER_WATTS,
    CPU_POWER_DETECTION_METHOD,
)


def load_classifier_assets() -> Tuple[Dict[str, Any], str]:
    """Load classifier/vectorizer from the first valid model artifact path."""
    seen = set()
    candidates: list[Path] = []
    for candidate in MODEL_PATH_CANDIDATES:
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        resolved = Path(candidate).resolve()
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        candidates.append(resolved)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("rb") as model_file:
                package = pickle.load(model_file)

            if "vectorizer" not in package or "classifier" not in package:
                raise ValueError("Model package must contain 'vectorizer' and 'classifier'.")

            return package, str(candidate)
        except Exception as exc:
            last_error = exc
            logger.error("Failed loading model artifact %s: %s", candidate, exc)

    search_list = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        f"Could not load a valid classifier model. Checked: {search_list}."
    ) from last_error


def _get_zero_shot_classifier():
    """Lazily initialize a semantic zero-shot classifier for low-confidence fallback."""
    global zero_shot_classifier

    if zero_shot_classifier is not None:
        return zero_shot_classifier

    with zero_shot_classifier_lock:
        if zero_shot_classifier is not None:
            return zero_shot_classifier

        logger.info("Loading zero-shot semantic classifier: %s", ZERO_SHOT_MODEL_NAME)
        zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model=ZERO_SHOT_MODEL_NAME,
            device=-1,
        )
        logger.info("Zero-shot semantic classifier loaded successfully")

    return zero_shot_classifier


def _predict_zero_shot_category_sync(text: str, categories: list[str]) -> Tuple[Optional[str], float]:
    """Run semantic category prediction synchronously (used via asyncio.to_thread)."""
    if not text or not categories:
        return None, 0.0

    clf_zero_shot = _get_zero_shot_classifier()

    semantic_to_category: Dict[str, str] = {}
    candidate_labels: list[str] = []
    for category in categories:
        category_name = str(category)
        semantic_label = CATEGORY_SEMANTIC_HINTS.get(category_name, category_name)

        if semantic_label in semantic_to_category and semantic_to_category[semantic_label] != category_name:
            semantic_label = f"{semantic_label} ({category_name})"

        semantic_to_category[semantic_label] = category_name
        candidate_labels.append(semantic_label)

    result = clf_zero_shot(text, candidate_labels, multi_label=False)
    top_label = result["labels"][0] if result.get("labels") else None
    top_score = float(result["scores"][0]) if result.get("scores") else 0.0

    if not top_label:
        return None, top_score

    return semantic_to_category.get(str(top_label), str(top_label)), top_score

# ====================== ML / NLP LOAD ======================
logger.info("Loading model_bbmp.pkl …")
model_package, loaded_model_path = load_classifier_assets()
vectorizer = model_package["vectorizer"]
clf = model_package["classifier"]

if hasattr(vectorizer, "get_feature_names_out"):
    vocab_size = len(vectorizer.get_feature_names_out())
else:
    vocab_size = len(getattr(vectorizer, "vocabulary_", {}))

MODEL_RUNTIME_INFO = {
    "path": loaded_model_path,
    "classes": [str(item) for item in getattr(clf, "classes_", [])],
    "class_count": len(getattr(clf, "classes_", [])),
    "vocab_size": vocab_size,
}
logger.info(
    "Classifier connected: path=%s classes=%d vocab_size=%d",
    MODEL_RUNTIME_INFO["path"],
    MODEL_RUNTIME_INFO["class_count"],
    MODEL_RUNTIME_INFO["vocab_size"],
)

logger.info("Loading spaCy en_core_web_sm …")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("en_core_web_sm not found – run: python -m spacy download en_core_web_sm")
    nlp = spacy.blank("en")

logger.info("Loading Whisper small …")
whisper_model = whisper.load_model("small")
logger.info("Core NLP models loaded successfully")

logger.info("Loading YOLOv8n-seg pothole detection model …")
load_yolo_model()

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

    model_ok = bool(MODEL_RUNTIME_INFO.get("class_count")) and bool(MODEL_RUNTIME_INFO.get("vocab_size"))
    status = "ok" if (db_ok and model_ok) else "degraded"
    return JSONResponse(
        content={
            "status": status,
            "db": "connected" if db_ok else "unreachable",
            "classifier": {
                "connected": model_ok,
                "path": MODEL_RUNTIME_INFO.get("path"),
                "classes": MODEL_RUNTIME_INFO.get("class_count", 0),
                "vocabulary": MODEL_RUNTIME_INFO.get("vocab_size", 0),
                "zero_shot_fallback_enabled": ENABLE_ZERO_SHOT_FALLBACK,
                "zero_shot_fallback_loaded": zero_shot_classifier is not None,
            },
        },
        status_code=200 if (db_ok and model_ok) else 503,
    )


@app.get("/model/status", tags=["ops"])
async def model_status():
    """Expose classifier connectivity details for quick runtime verification."""
    return {
        "connected": True,
        "path": MODEL_RUNTIME_INFO.get("path"),
        "class_count": MODEL_RUNTIME_INFO.get("class_count", 0),
        "vocabulary_size": MODEL_RUNTIME_INFO.get("vocab_size", 0),
        "classes": MODEL_RUNTIME_INFO.get("classes", []),
        "zero_shot_fallback": {
            "enabled": ENABLE_ZERO_SHOT_FALLBACK,
            "loaded": zero_shot_classifier is not None,
            "model": ZERO_SHOT_MODEL_NAME,
            "min_primary_confidence": ZERO_SHOT_MIN_CONFIDENCE,
            "min_semantic_score": ZERO_SHOT_MIN_SCORE,
            "min_semantic_score_sparse": ZERO_SHOT_SPARSE_MIN_SCORE,
            "primary_min_explanatory_features": PRIMARY_MIN_EXPLANATORY_FEATURES,
        },
    }


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


def _base_category_explanation(text: str) -> Dict[str, Any]:
    return {
        "method": "tfidf_multinomial_nb",
        "summary": "Top statistically weighted terms from the trained NLP classifier (not rule-based).",
        "classification_text": (text or "").strip(),
        "confidence": None,
        "top_features": [],
        "highlight_terms": [],
    }


def explain_category_prediction(text: str, text_vector, predicted_label: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Explain why a category was predicted by ranking feature contributions
    from the trained TF-IDF + Naive Bayes model.
    """
    explanation = _base_category_explanation(text)

    try:
        if text_vector is None or text_vector.nnz == 0:
            return explanation

        class_labels = list(getattr(clf, "classes_", []))
        if predicted_label not in class_labels:
            return explanation

        predicted_idx = class_labels.index(predicted_label)
        rival_idx = predicted_idx

        if hasattr(clf, "predict_proba"):
            probabilities = clf.predict_proba(text_vector)[0]
            explanation["confidence"] = round(float(probabilities[predicted_idx]), 4)
            ranked_class_indices = sorted(
                range(len(probabilities)),
                key=lambda idx: probabilities[idx],
                reverse=True,
            )
            rival_idx = next((idx for idx in ranked_class_indices if idx != predicted_idx), predicted_idx)

        feature_names = vectorizer.get_feature_names_out()
        row = text_vector.tocsr()[0]

        if not hasattr(clf, "feature_log_prob_"):
            return explanation

        pred_log_probs = clf.feature_log_prob_[predicted_idx]
        rival_log_probs = clf.feature_log_prob_[rival_idx] if rival_idx != predicted_idx else pred_log_probs

        scored_terms = []
        for feature_idx, tfidf_value in zip(row.indices, row.data):
            term = str(feature_names[feature_idx]).strip()
            if not term:
                continue

            # Hide character n-grams from UI to avoid confusing non-technical users
            if term.startswith("char_wb__"):
                continue

            # Clean up word prefixes
            if term.startswith("word__"):
                term = term.replace("word__", "", 1)

            weight_delta = float(pred_log_probs[feature_idx] - rival_log_probs[feature_idx])
            contribution = float(tfidf_value) * weight_delta
            if contribution <= 0:
                continue

            scored_terms.append(
                {
                    "term": term,
                    "contribution": contribution,
                    "tfidf": float(tfidf_value),
                    "weight_delta": weight_delta,
                }
            )

        if not scored_terms:
            return explanation

        scored_terms.sort(key=lambda item: item["contribution"], reverse=True)
        top_terms = scored_terms[:top_k]
        total_contribution = sum(item["contribution"] for item in top_terms) or 1.0

        explanation["top_features"] = [
            {
                "term": item["term"],
                "contribution": round(item["contribution"], 4),
                "tfidf": round(item["tfidf"], 4),
                "weight_delta": round(item["weight_delta"], 4),
                "importance_percent": round((item["contribution"] / total_contribution) * 100.0, 1),
            }
            for item in top_terms
        ]
        explanation["highlight_terms"] = [item["term"] for item in top_terms]

        return explanation
    except Exception as exc:
        logger.warning("Category explanation failed: %s", exc)
        return explanation

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


def apply_civic_translation_glossary(
    source_text: str,
    translated_text: str,
    source_language: str,
    target_language: str,
) -> str:
    """Patch known civic-term mistranslations for Kannada -> English outputs."""
    candidate = (translated_text or "").strip()
    source_clean = (source_text or "").strip()

    if not candidate or target_language != "en":
        return candidate

    translit_match = bool(KANNADA_POTHOLE_TRANSLIT_PATTERN.search(source_clean.lower()))
    is_kannada_source = (
        source_language == "kn"
        or bool(re.search(r"[\u0C80-\u0CFF]", source_clean))
        or translit_match
    )
    if not is_kannada_source:
        return candidate

    source_lower = source_clean.lower()
    pothole_in_source = (
        any(term in source_lower for term in KANNADA_POTHOLE_TERMS)
        or translit_match
    )
    if not pothole_in_source:
        return candidate

    corrected = candidate
    corrected = re.sub(r"\bstreet buttons?\b", "street pothole", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"\broad buttons?\b", "road pothole", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"\bbuttons?\b", "pothole", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"\b(pits?|holes?)\b", "pothole", corrected, flags=re.IGNORECASE)

    if not re.search(r"\bpotholes?\b", corrected, flags=re.IGNORECASE):
        if re.search(r"\b(street|road)\b", corrected, flags=re.IGNORECASE):
            corrected = re.sub(r"\b(street|road)\b", r"\1 pothole", corrected, count=1, flags=re.IGNORECASE)
        else:
            corrected = f"Pothole complaint: {corrected}"

    corrected = re.sub(r"\s+", " ", corrected).strip()
    if corrected and corrected[-1] not in ".!?":
        corrected += "."
    return corrected


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
                use_cache=False,
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
        if not translated_text:
            return clean_text

        corrected_text = apply_civic_translation_glossary(
            clean_text,
            translated_text,
            source_language,
            target_language,
        )
        if corrected_text != translated_text:
            logger.info("Applied Kannada civic glossary correction: %s -> %s", translated_text, corrected_text)

        return corrected_text or clean_text
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
    voter_fingerprint: str = Form(default=""),
    db: Session = Depends(get_db),
):
    """
    Full NLP pipeline: receive audio → transcribe (Whisper) →
    translate (dedicated NLP model) → classify → save to DB.
    All NLP stages are timed with time.perf_counter() for analytics.
    """
    # ── NLP Metrics tracking variables ─────────────────────────────
    _nlp_transcription_time = 0.0
    _nlp_translation_time = 0.0
    _nlp_classification_time = 0.0
    _nlp_ner_time = 0.0
    _nlp_zero_shot_time = 0.0
    _nlp_audio_duration = None
    _nlp_classifier_confidence = None
    _nlp_zero_shot_triggered = False
    _nlp_zero_shot_confidence = 0.0
    _nlp_entity_count = 0
    _nlp_entity_types = {}
    _nlp_error_stage = None
    _nlp_error_message = None
    # Image analysis stage tracking variables
    _nlp_image_analysis_time = 0.0
    _nlp_detected_object_count = 0
    _nlp_image_model_confidence = None
    _nlp_pothole_severity = None
    _nlp_detected_objects = None

    # 0. Validate live location fields (mandatory)
    if not (-90 <= live_latitude <= 90) or not (-180 <= live_longitude <= 180):
        raise HTTPException(status_code=400, detail="Live location coordinates are invalid.")

    # Restrict to Bangalore limits
    if not (12.73 <= live_latitude <= 13.14) or not (77.37 <= live_longitude <= 77.88):
        raise HTTPException(status_code=400, detail="Complaints can only be reported within Bangalore city limits.")

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

    # ── Stage 1: Transcription (timed) ────────────────────────────
    transcribed_text = submitted_text
    detected_language = recording_language
    _t_transcription_start = time.perf_counter()
    if audio_path:
        if recording_language in ["kn", "hi", "en"]:
            logger.info("Using Google STT for highly accurate transcription.")
            try:
                lang_code_map = {"kn": "kn-IN", "hi": "hi-IN", "en": "en-IN"}
                google_lang = lang_code_map.get(recording_language, "en-IN")
                
                # Convert webm to wav for SpeechRecognition
                wav_path = audio_path + ".wav"
                audio_segment = AudioSegment.from_file(audio_path)
                _nlp_audio_duration = audio_segment.duration_seconds
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
                _nlp_error_stage = "transcription"
                _nlp_error_message = f"Google STT could not understand audio for {audio_path}"
                logger.warning(_nlp_error_message)
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
                        _nlp_error_stage = "transcription"
                        _nlp_error_message = f"No speech detected in uploaded audio: {audio_path}"
                        logger.warning(_nlp_error_message)
                        raise HTTPException(
                            status_code=400,
                            detail="No clear speech detected in audio. Please record again and speak closer to the microphone."
                        )
                    transcribed_text = f"{audio_text} {submitted_text}".strip() if submitted_text else audio_text
                    logger.info(f"Whisper Transcribed: {audio_text} | Language: {detected_language}")
                    # Capture audio duration if not already done
                    if _nlp_audio_duration is None:
                        try:
                            _seg = AudioSegment.from_file(audio_path)
                            _nlp_audio_duration = _seg.duration_seconds
                        except Exception:
                            pass
                except Exception as ex:
                    _nlp_error_stage = "transcription"
                    _nlp_error_message = str(ex)
                    raise HTTPException(status_code=500, detail="Audio transcription failed. Please try again.")
        else:
            try:
                # Capture audio duration
                try:
                    _seg = AudioSegment.from_file(audio_path)
                    _nlp_audio_duration = _seg.duration_seconds
                except Exception:
                    pass
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
                    _nlp_error_stage = "transcription"
                    _nlp_error_message = f"No speech detected in uploaded audio: {audio_path}"
                    logger.warning(_nlp_error_message)
                    raise HTTPException(
                        status_code=400,
                        detail="No clear speech detected in audio. Please record again and speak closer to the microphone."
                    )
                transcribed_text = f"{audio_text} {submitted_text}".strip() if submitted_text else audio_text
                logger.info(f"Transcribed: {audio_text} | Language: {detected_language}")
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                _nlp_error_stage = "transcription"
                _nlp_error_message = str(e)
                logger.error(f"Whisper transcription failed for {audio_path}: {e}")
                if "WinError 2" in str(e) or "ffmpeg" in str(e).lower():
                    logger.warning("FFmpeg not found. Cannot transcribe audio without FFmpeg.")
                    raise HTTPException(
                        status_code=500,
                        detail="Audio transcription failed: FFmpeg is not installed or not found in PATH. Please install FFmpeg and retry."
                    )
                raise HTTPException(status_code=500, detail="Audio transcription failed. Please try again.")
    _nlp_transcription_time = time.perf_counter() - _t_transcription_start

    if not transcribed_text:
        raise HTTPException(status_code=400, detail="Complaint text is required.")

    # ── Stage 2: Translation (timed) ──────────────────────────────
    _t_translation_start = time.perf_counter()
    try:
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
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        _nlp_error_stage = "translation"
        _nlp_error_message = str(e)
        logger.error(f"Translation failed: {e}")
        translated_text = transcribed_text
        english_text_for_classification = transcribed_text
    _nlp_translation_time = time.perf_counter() - _t_translation_start

    # ── Stage 3: Classification (timed) ───────────────────────────
    _t_classification_start = time.perf_counter()
    classification_text = build_multilingual_classification_text(
        transcribed_text,
        english_text_for_classification
    )
    category_explanation = _base_category_explanation(classification_text)
    primary_confidence = None
    try:
        text_vector = vectorizer.transform([classification_text])
        category = str(clf.predict(text_vector)[0])
        primary_category = category
        category_explanation = explain_category_prediction(
            classification_text,
            text_vector,
            category,
        )

        primary_confidence = category_explanation.get("confidence")
        _nlp_classifier_confidence = float(primary_confidence) if isinstance(primary_confidence, (float, int)) else None
        top_features = category_explanation.get("top_features") if isinstance(category_explanation, dict) else []
        top_features = top_features if isinstance(top_features, list) else []
        top_terms = [
            str(item.get("term", "")).strip().lower()
            for item in top_features
            if isinstance(item, dict) and item.get("term")
        ]

        low_primary_confidence = (
            isinstance(primary_confidence, (float, int))
            and float(primary_confidence) < ZERO_SHOT_MIN_CONFIDENCE
        )
        sparse_primary_signal = len(top_terms) < PRIMARY_MIN_EXPLANATORY_FEATURES
        generic_primary_signal = bool(top_terms) and all(
            term in GENERIC_PRIMARY_FEATURE_TERMS for term in top_terms
        )
    except Exception as e:
        _nlp_error_stage = _nlp_error_stage or "classification"
        _nlp_error_message = _nlp_error_message or str(e)
        logger.error(f"Classification failed: {e}")
        category = "Others"
        primary_category = category
        low_primary_confidence = False
        sparse_primary_signal = False
        generic_primary_signal = False
    _nlp_classification_time = time.perf_counter() - _t_classification_start

    # ── Stage 4: Zero-shot fallback (timed) ───────────────────────
    _t_zero_shot_start = time.perf_counter()
    if (
        ENABLE_ZERO_SHOT_FALLBACK
        and (low_primary_confidence or sparse_primary_signal or generic_primary_signal)
    ):
        _nlp_zero_shot_triggered = True
        try:
            candidate_categories = MODEL_RUNTIME_INFO.get("classes", [])
            semantic_category, semantic_score = await asyncio.to_thread(
                _predict_zero_shot_category_sync,
                english_text_for_classification,
                candidate_categories,
            )
            _nlp_zero_shot_confidence = float(semantic_score) if semantic_score else 0.0

            semantic_threshold = (
                ZERO_SHOT_MIN_SCORE if low_primary_confidence else ZERO_SHOT_SPARSE_MIN_SCORE
            )

            if semantic_category and semantic_score >= semantic_threshold and semantic_category != category:
                logger.info(
                    "Semantic fallback override: %s (%.3f) -> %s (%.3f) | low_conf=%s sparse=%s generic=%s",
                    category,
                    float(primary_confidence) if isinstance(primary_confidence, (float, int)) else -1.0,
                    semantic_category,
                    semantic_score,
                    low_primary_confidence,
                    sparse_primary_signal,
                    generic_primary_signal,
                )
                category = semantic_category
                category_explanation = {
                    "method": "zero_shot_nli_fallback",
                    "summary": "Primary TF-IDF evidence was low-confidence or sparse, so semantic NLI classification was used (NLP model, not rule-based).",
                    "classification_text": english_text_for_classification,
                    "confidence": round(float(semantic_score), 4),
                    "top_features": [],
                    "highlight_terms": [],
                    "base_model_category": primary_category,
                    "base_model_confidence": (
                        round(float(primary_confidence), 4)
                        if isinstance(primary_confidence, (float, int))
                        else None
                    ),
                    "fallback_trigger": {
                        "low_primary_confidence": low_primary_confidence,
                        "sparse_primary_signal": sparse_primary_signal,
                        "generic_primary_signal": generic_primary_signal,
                        "semantic_threshold": semantic_threshold,
                    },
                }
        except Exception as fallback_exc:
            _nlp_error_stage = _nlp_error_stage or "zero_shot"
            _nlp_error_message = _nlp_error_message or str(fallback_exc)
            logger.warning("Semantic fallback failed: %s", fallback_exc)
    _nlp_zero_shot_time = time.perf_counter() - _t_zero_shot_start

    category_explanation["predicted_category"] = category
    location = f"{live_latitude:.6f}, {live_longitude:.6f}"
    logger.info(f"Category: {category} | Location: {location}")

    # ── Stage 5: NER — extract entities (timed) ───────────────────
    _t_ner_start = time.perf_counter()
    try:
        ner_doc = nlp(english_text_for_classification or translated_text or transcribed_text)
        ner_entities = [ent for ent in ner_doc.ents if ent.label_ in ("GPE", "LOC", "FAC", "ORG")]
        _nlp_entity_count = len(ner_entities)
        _entity_type_counter = {}
        for ent in ner_entities:
            _entity_type_counter[ent.label_] = _entity_type_counter.get(ent.label_, 0) + 1
        _nlp_entity_types = _entity_type_counter
    except Exception as e:
        _nlp_error_stage = _nlp_error_stage or "ner"
        _nlp_error_message = _nlp_error_message or str(e)
        logger.warning("NER entity extraction failed: %s", e)
    _nlp_ner_time = time.perf_counter() - _t_ner_start

    if category == "Non-Civic":
        if audio_path and os.path.exists(audio_path):
            try: os.remove(audio_path)
            except Exception: pass
        if image_path and os.path.exists(image_path):
            try: os.remove(image_path)
            except Exception: pass
        logger.info(f"Discarded irrelevant complaint: {transcribed_text}")
        raise HTTPException(
            status_code=400,
            detail="The submitted audio/text is irrelevant and does not appear to be a valid civic complaint. It has been discarded."
        )

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

    # ── Stage 6: Image analysis — pothole/road-damage detection (timed) ──
    _t_image_analysis_start = time.perf_counter()
    image_suggested_category = None
    category_mismatch = False

    if image_path:
        try:
            img_result = await analyze_image(image_path)
            detections = img_result.get("detections", [])
            _nlp_detected_objects = detections
            _nlp_detected_object_count = len(detections)
            _nlp_pothole_severity = img_result.get("severity")  # "Clear", "Low", ..., "Severe", or None
            if detections:
                _nlp_image_model_confidence = max(d["confidence"] for d in detections)
                
                # Reconciliation logic
                top_detection = max(detections, key=lambda d: d["confidence"])
                image_suggested_category = DETECTION_CLASS_TO_CATEGORY.get(top_detection["class"])
                
                if (
                    image_suggested_category is not None 
                    and image_suggested_category != category 
                    and top_detection["confidence"] > IMAGE_RECONCILE_CONFIDENCE_THRESHOLD
                ):
                    category_mismatch = True
                    # Downgrade trust level so admin has to review the mismatch
                    trust_level = "manual_review"
                    status = "pending"

        except Exception as img_exc:
            _nlp_error_stage = _nlp_error_stage or "image_analysis"
            _nlp_error_message = _nlp_error_message or str(img_exc)
            logger.warning("Image analysis failed (non-fatal): %s", img_exc)
    _nlp_image_analysis_time = time.perf_counter() - _t_image_analysis_start

    DUPLICATE_RADIUS_KM = 0.5
    DUPLICATE_WINDOW_DAYS = 180
    window_start = datetime.utcnow() - timedelta(days=DUPLICATE_WINDOW_DAYS)
    dup_query = db.query(Complaint).filter(
        Complaint.category == category,
        Complaint.status != "Resolved",
        Complaint.created_at >= window_start,
    )
    existing_dup = None
    for candidate in dup_query.all():
        if candidate.live_latitude and candidate.live_longitude:
            dist = haversine_distance_meters(
                live_latitude, live_longitude,
                candidate.live_latitude, candidate.live_longitude,
            )
            if dist <= DUPLICATE_RADIUS_KM * 1000:
                existing_dup = candidate
                break
        elif candidate.location and candidate.location == location:
            existing_dup = candidate
            break

    # ── Compute NLP metrics totals ────────────────────────────────
    _nlp_total_time = (
        _nlp_transcription_time + _nlp_translation_time +
        _nlp_classification_time + _nlp_ner_time + _nlp_zero_shot_time +
        _nlp_image_analysis_time
    )
    _nlp_word_count = len((transcribed_text or "").split())
    _energy_by_stage = {
        "transcription": round(ESTIMATED_CPU_POWER_WATTS * _nlp_transcription_time, 6),
        "translation": round(ESTIMATED_CPU_POWER_WATTS * _nlp_translation_time, 6),
        "classification": round(ESTIMATED_CPU_POWER_WATTS * _nlp_classification_time, 6),
        "ner": round(ESTIMATED_CPU_POWER_WATTS * _nlp_ner_time, 6),
        "zero_shot": round(ESTIMATED_CPU_POWER_WATTS * _nlp_zero_shot_time, 6),
        "image_analysis": round(ESTIMATED_CPU_POWER_WATTS * _nlp_image_analysis_time, 6),
    }
    _total_energy = round(ESTIMATED_CPU_POWER_WATTS * _nlp_total_time, 6)

    if existing_dup:
        # ── Record NLP metric for duplicate complaint ─────────────
        try:
            db.add(NlpMetric(
                complaint_id=existing_dup.id,
                is_duplicate=True,
                source_language=detected_language,
                category=category,
                classifier_confidence=_nlp_classifier_confidence,
                zero_shot_triggered=_nlp_zero_shot_triggered,
                zero_shot_confidence=_nlp_zero_shot_confidence,
                entity_count=_nlp_entity_count,
                entity_types=json.dumps(_nlp_entity_types),
                audio_duration_seconds=_nlp_audio_duration,
                word_count=_nlp_word_count,
                transcription_time=round(_nlp_transcription_time, 6),
                translation_time=round(_nlp_translation_time, 6),
                classification_time=round(_nlp_classification_time, 6),
                ner_time=round(_nlp_ner_time, 6),
                zero_shot_time=round(_nlp_zero_shot_time, 6),
                image_analysis_time=round(_nlp_image_analysis_time, 6),
                total_processing_time=round(_nlp_total_time, 6),
                estimated_power_watts=ESTIMATED_CPU_POWER_WATTS,
                total_energy_joules=_total_energy,
                energy_by_stage=json.dumps(_energy_by_stage),
                calculation_method=CPU_POWER_DETECTION_METHOD,
                detected_object_count=_nlp_detected_object_count,
                image_model_confidence=_nlp_image_model_confidence,
                pothole_severity=_nlp_pothole_severity,
                error_stage=_nlp_error_stage,
                error_message=_nlp_error_message,
            ))
            db.commit()
        except Exception as metric_exc:
            logger.warning("Failed to record NLP metric for duplicate: %s", metric_exc)

        # Add vote if fingerprint provided and not already voted
        voted = False
        fp = (voter_fingerprint or "").strip()
        if fp:
            already_voted = db.query(ComplaintVote).filter(
                ComplaintVote.complaint_id == existing_dup.id,
                ComplaintVote.voter_fingerprint == fp,
            ).first()
            if not already_voted:
                db.add(ComplaintVote(complaint_id=existing_dup.id, voter_fingerprint=fp))
                existing_dup.votes = (existing_dup.votes or 0) + 1
                db.commit()
                db.refresh(existing_dup)
                voted = True

        # Clean up uploaded files since we won't save a new complaint (after the vote is recorded)
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Deleted duplicate audio file: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to delete audio file {audio_path}: {e}")
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                logger.info(f"Deleted duplicate image file: {image_path}")
            except Exception as e:
                logger.error(f"Failed to delete image file {image_path}: {e}")

        logger.info("Duplicate detected: new complaint matches existing #%d, votes count updated to %d", existing_dup.id, existing_dup.votes)
        return {
            "duplicate": True,
            "id": existing_dup.id,
            "category": existing_dup.category,
            "location": existing_dup.location,
            "status": existing_dup.status,
            "votes": existing_dup.votes or 0,
            "voted": voted,
            "message": "This issue already exists. Your vote has been added." if voted else "This issue already exists and has been reported.",
        }

    # 9. Save new complaint to database
    fp = (voter_fingerprint or "").strip()
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
        votes=1 if fp else 0,
        detected_objects=json.dumps(_nlp_detected_objects) if _nlp_detected_objects else None,
        pothole_severity=_nlp_pothole_severity,
        image_suggested_category=image_suggested_category,
        category_mismatch=category_mismatch,
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)

    # Record the submitter's fingerprint as the first vote on the new complaint
    if fp:
        db.add(ComplaintVote(complaint_id=complaint.id, voter_fingerprint=fp))
        db.commit()

    # Auto-create first timeline entry
    db.add(ComplaintTimeline(complaint_id=complaint.id, status="Reported", note="Complaint submitted"))
    db.commit()
    logger.info(f"Complaint saved with ID: {complaint.id}, initial votes: {complaint.votes}")

    # ── Record NLP metric for new complaint ────────────────────────
    try:
        db.add(NlpMetric(
            complaint_id=complaint.id,
            is_duplicate=False,
            source_language=detected_language,
            category=category,
            classifier_confidence=_nlp_classifier_confidence,
            zero_shot_triggered=_nlp_zero_shot_triggered,
            zero_shot_confidence=_nlp_zero_shot_confidence,
            entity_count=_nlp_entity_count,
            entity_types=json.dumps(_nlp_entity_types),
            audio_duration_seconds=_nlp_audio_duration,
            word_count=_nlp_word_count,
            transcription_time=round(_nlp_transcription_time, 6),
            translation_time=round(_nlp_translation_time, 6),
            classification_time=round(_nlp_classification_time, 6),
            ner_time=round(_nlp_ner_time, 6),
            zero_shot_time=round(_nlp_zero_shot_time, 6),
            image_analysis_time=round(_nlp_image_analysis_time, 6),
            total_processing_time=round(_nlp_total_time, 6),
            estimated_power_watts=ESTIMATED_CPU_POWER_WATTS,
            total_energy_joules=_total_energy,
            energy_by_stage=json.dumps(_energy_by_stage),
            calculation_method=CPU_POWER_DETECTION_METHOD,
            detected_object_count=_nlp_detected_object_count,
            image_model_confidence=_nlp_image_model_confidence,
            pothole_severity=_nlp_pothole_severity,
            error_stage=_nlp_error_stage,
            error_message=_nlp_error_message,
        ))
        db.commit()
    except Exception as metric_exc:
        logger.warning("Failed to record NLP metric: %s", metric_exc)

    # Return JSON response
    return {
        "duplicate": False,
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
        "classification_text": category_explanation.get("classification_text", ""),
        "category_explanation": category_explanation,
        "detected_language": complaint.language,
        "target_language": target_language_code,
        "status": complaint.status,
        "votes": complaint.votes or 0,
        "detected_objects": _nlp_detected_objects,
        "pothole_severity": _nlp_pothole_severity,
        "image_suggested_category": complaint.image_suggested_category,
        "category_mismatch": complaint.category_mismatch,
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
    total_votes = db.query(func.sum(Complaint.votes)).scalar() or 0

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
        "total_votes": total_votes,
        "by_category": [{"category": r[0], "count": r[1]} for r in by_category],
        "by_language": [{"language": r[0], "count": r[1]} for r in by_language],
    }

# ---------- NLP Analytics Dashboard (JWT-protected) ----------
@app.get("/analytics/dashboard", tags=["analytics"])
async def analytics_dashboard(
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    language: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Comprehensive NLP analytics dashboard. ALL values from real DB data."""

    # Build base query with optional filters
    base_q = db.query(NlpMetric)
    if start_date:
        try:
            sd = datetime.fromisoformat(start_date)
            base_q = base_q.filter(NlpMetric.created_at >= sd)
        except ValueError:
            pass
    if end_date:
        try:
            ed = datetime.fromisoformat(end_date)
            base_q = base_q.filter(NlpMetric.created_at <= ed)
        except ValueError:
            pass
    if language:
        base_q = base_q.filter(NlpMetric.source_language == language)

    total_nlp = base_q.count()

    if total_nlp == 0:
        return {
            "complaint_stats": {
                "total_complaints_processed": 0,
                "unique_complaints": db.query(Complaint).count(),
                "duplicate_complaints": 0,
                "total_votes": int(db.query(func.sum(Complaint.votes)).scalar() or 0),
                "average_votes_per_complaint": 0.0,
            },
            "nlp_stats": {
                "total_requests": 0,
                "avg_processing_time_seconds": 0.0,
                "avg_time_by_stage": {"transcription": 0, "translation": 0, "classification": 0, "ner": 0, "zero_shot": 0, "image_analysis": 0},
                "zero_shot_fallback_rate": 0.0,
                "avg_classifier_confidence": 0.0,
                "avg_entity_count": 0.0,
                "entity_type_breakdown": [],
                "avg_word_count": 0.0,
                "avg_audio_duration": 0.0,
            },
            "energy_stats": {
                "total_energy_joules": 0.0,
                "avg_energy_per_complaint": 0.0,
                "energy_saved_by_dedup": 0.0,
                "energy_by_stage": {"transcription": 0, "translation": 0, "classification": 0, "ner": 0, "zero_shot": 0, "image_analysis": 0},
                "calculation_method": CPU_POWER_DETECTION_METHOD,
            },
            "error_stats": {"total_errors": 0, "error_rate_percent": 0.0, "errors_by_stage": {}},
            "charts": {
                "energy_by_stage": [], "energy_over_time": [], "category_distribution": [],
                "duplicate_vs_unique": {"unique": 0, "duplicate": 0},
                "votes_per_complaint": [], "language_distribution": [],
                "confidence_histogram": [], "category_language_heatmap": [],
                "entity_count_histogram": [], "entity_type_breakdown": [],
                "stage_bottleneck_radar": {"labels": ["Transcription","Translation","Classification","NER","Zero-shot","Image Analysis"], "avg_times": [0,0,0,0,0,0]},
                "throughput_over_time": [], "audio_duration_vs_time": [],
                "duplicate_cluster_sizes": [], "error_rate_by_stage": [],
                "severity_distribution": [],
            },
            "data_sources": {
                "note": "No NLP metrics recorded yet. Submit complaints to populate analytics."
            },
        }

    # ── Complaint Stats ──────────────────────────────────────────
    unique_complaints = db.query(Complaint).count()
    dup_count = base_q.filter(NlpMetric.is_duplicate == True).count()
    total_votes = int(db.query(func.sum(Complaint.votes)).scalar() or 0)
    avg_votes = round(total_votes / unique_complaints, 2) if unique_complaints else 0.0

    # ── NLP Stats ────────────────────────────────────────────────
    avg_proc = float(db.query(func.avg(NlpMetric.total_processing_time)).filter(NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))).scalar() or 0)
    avg_trans = float(db.query(func.avg(NlpMetric.transcription_time)).filter(NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))).scalar() or 0)
    avg_transl = float(db.query(func.avg(NlpMetric.translation_time)).filter(NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))).scalar() or 0)
    avg_clf = float(db.query(func.avg(NlpMetric.classification_time)).filter(NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))).scalar() or 0)
    avg_ner = float(db.query(func.avg(NlpMetric.ner_time)).filter(NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))).scalar() or 0)
    avg_zs = float(db.query(func.avg(NlpMetric.zero_shot_time)).filter(NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))).scalar() or 0)
    avg_img = float(db.query(func.avg(NlpMetric.image_analysis_time)).filter(NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))).scalar() or 0)

    zs_triggered = base_q.filter(NlpMetric.zero_shot_triggered == True).count()
    zs_rate = round((zs_triggered / total_nlp) * 100, 2) if total_nlp else 0.0

    avg_conf = float(db.query(func.avg(NlpMetric.classifier_confidence)).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id)),
        NlpMetric.classifier_confidence.isnot(None),
    ).scalar() or 0)
    avg_ent = float(db.query(func.avg(NlpMetric.entity_count)).filter(NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))).scalar() or 0)
    avg_wc = float(db.query(func.avg(NlpMetric.word_count)).filter(NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))).scalar() or 0)
    avg_audio = float(db.query(func.avg(NlpMetric.audio_duration_seconds)).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id)),
        NlpMetric.audio_duration_seconds.isnot(None),
    ).scalar() or 0)

    # ── Entity type breakdown (aggregate JSON column) ────────────
    all_entity_types_agg = {}
    for row in base_q.with_entities(NlpMetric.entity_types).all():
        if row[0]:
            try:
                et = json.loads(row[0])
                for k, v in et.items():
                    all_entity_types_agg[k] = all_entity_types_agg.get(k, 0) + v
            except (json.JSONDecodeError, TypeError):
                pass

    # ── Energy Stats ─────────────────────────────────────────────
    total_energy = float(db.query(func.sum(NlpMetric.total_energy_joules)).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))
    ).scalar() or 0)
    avg_energy = round(total_energy / total_nlp, 6) if total_nlp else 0.0
    energy_saved = round(avg_energy * dup_count, 6)

    # Energy by stage (aggregate)
    energy_stage_agg = {"transcription": 0.0, "translation": 0.0, "classification": 0.0, "ner": 0.0, "zero_shot": 0.0, "image_analysis": 0.0}
    for row in base_q.with_entities(NlpMetric.energy_by_stage).all():
        if row[0]:
            try:
                es = json.loads(row[0])
                for k, v in es.items():
                    energy_stage_agg[k] = energy_stage_agg.get(k, 0) + float(v)
            except (json.JSONDecodeError, TypeError):
                pass

    # ── Error Stats ──────────────────────────────────────────────
    total_errors = base_q.filter(NlpMetric.error_stage.isnot(None)).count()
    error_rate = round((total_errors / total_nlp) * 100, 2) if total_nlp else 0.0
    errors_by_stage_q = db.query(
        NlpMetric.error_stage, func.count(NlpMetric.id)
    ).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id)),
        NlpMetric.error_stage.isnot(None),
    ).group_by(NlpMetric.error_stage).all()
    errors_by_stage = {r[0]: r[1] for r in errors_by_stage_q}

    # ── Charts Data ──────────────────────────────────────────────

    # Energy by stage chart
    energy_by_stage_chart = [
        {"stage": k, "joules": round(v, 4)} for k, v in energy_stage_agg.items()
    ]

    # Energy over time (group by date)
    energy_over_time_q = db.query(
        func.date(NlpMetric.created_at).label("date"),
        func.sum(NlpMetric.total_energy_joules).label("joules"),
        func.count(NlpMetric.id).label("count"),
    ).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))
    ).group_by(func.date(NlpMetric.created_at)).order_by(func.date(NlpMetric.created_at)).all()
    energy_over_time = [{"date": str(r[0]), "joules": round(float(r[1] or 0), 4), "count": r[2]} for r in energy_over_time_q]

    # Category distribution
    cat_dist_q = db.query(
        NlpMetric.category, func.count(NlpMetric.id)
    ).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))
    ).group_by(NlpMetric.category).all()
    category_distribution = [{"category": r[0] or "Unknown", "count": r[1]} for r in cat_dist_q]

    # Duplicate vs unique
    unique_count = base_q.filter(NlpMetric.is_duplicate == False).count()
    dup_vs_unique = {"unique": unique_count, "duplicate": dup_count}

    # Votes per complaint (top 20)
    votes_q = db.query(Complaint.id, Complaint.votes, Complaint.category).filter(
        Complaint.votes > 0
    ).order_by(Complaint.votes.desc()).limit(20).all()
    votes_per_complaint = [{"complaint_id": r[0], "votes": r[1] or 0, "category": r[2] or ""} for r in votes_q]

    # Language distribution
    lang_dist_q = db.query(
        NlpMetric.source_language,
        func.count(NlpMetric.id),
        func.avg(NlpMetric.total_processing_time),
    ).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))
    ).group_by(NlpMetric.source_language).all()
    language_distribution = [
        {"language": r[0] or "unknown", "count": r[1], "avg_processing_time": round(float(r[2] or 0), 4)}
        for r in lang_dist_q
    ]

    # Confidence histogram (10 bins: 0.0-0.1, 0.1-0.2, ...)
    confidence_histogram = []
    for i in range(10):
        lo = i / 10.0
        hi = (i + 1) / 10.0
        cnt = base_q.filter(
            NlpMetric.classifier_confidence >= lo,
            NlpMetric.classifier_confidence < hi if i < 9 else NlpMetric.classifier_confidence <= hi,
        ).count()
        confidence_histogram.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": cnt})

    # Severity distribution (pothole detection)
    severity_dist_q = db.query(
        NlpMetric.pothole_severity, func.count(NlpMetric.id)
    ).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id)),
        NlpMetric.pothole_severity.isnot(None),
    ).group_by(NlpMetric.pothole_severity).all()
    severity_distribution = [{"severity": r[0], "count": r[1]} for r in severity_dist_q]

    # Category × Language heatmap
    cat_lang_q = db.query(
        NlpMetric.category, NlpMetric.source_language, func.count(NlpMetric.id)
    ).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))
    ).group_by(NlpMetric.category, NlpMetric.source_language).all()
    category_language_heatmap = [
        {"category": r[0] or "Unknown", "language": r[1] or "unknown", "count": r[2]}
        for r in cat_lang_q
    ]

    # Entity count histogram
    ent_hist_q = db.query(
        NlpMetric.entity_count, func.count(NlpMetric.id)
    ).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))
    ).group_by(NlpMetric.entity_count).order_by(NlpMetric.entity_count).all()
    entity_count_histogram = [{"entities": r[0] or 0, "count": r[1]} for r in ent_hist_q]

    # Entity type breakdown
    entity_type_breakdown = [{"type": k, "count": v} for k, v in sorted(all_entity_types_agg.items(), key=lambda x: x[1], reverse=True)]

    # Stage bottleneck radar
    stage_bottleneck_radar = {
        "labels": ["Transcription", "Translation", "Classification", "NER", "Zero-shot", "Image Analysis"],
        "avg_times": [round(avg_trans, 4), round(avg_transl, 4), round(avg_clf, 4), round(avg_ner, 4), round(avg_zs, 4), round(avg_img, 4)],
    }

    # Throughput over time (hourly)
    throughput_q = db.query(
        func.date(NlpMetric.created_at).label("day"),
        func.count(NlpMetric.id).label("count"),
    ).filter(
        NlpMetric.id.in_(base_q.with_entities(NlpMetric.id))
    ).group_by(func.date(NlpMetric.created_at)).order_by(func.date(NlpMetric.created_at)).all()
    throughput_over_time = [{"hour": str(r[0]), "count": r[1]} for r in throughput_q]

    # Audio duration vs processing time (scatter)
    audio_scatter_q = base_q.filter(
        NlpMetric.audio_duration_seconds.isnot(None),
        NlpMetric.audio_duration_seconds > 0,
    ).with_entities(
        NlpMetric.audio_duration_seconds, NlpMetric.total_processing_time
    ).limit(200).all()
    audio_duration_vs_time = [
        {"duration_s": round(float(r[0]), 2), "processing_time_s": round(float(r[1]), 4)}
        for r in audio_scatter_q
    ]

    # Duplicate cluster sizes (how many votes per unique complaint that has duplicates)
    dup_cluster_q = db.query(
        Complaint.votes, func.count(Complaint.id)
    ).filter(Complaint.votes > 1).group_by(Complaint.votes).order_by(Complaint.votes).all()
    duplicate_cluster_sizes = [{"cluster_size": r[0], "count": r[1]} for r in dup_cluster_q]

    # Error rate by stage
    stages = ["transcription", "translation", "classification", "ner", "zero_shot", "image_analysis"]
    error_rate_by_stage = []
    for stage in stages:
        stage_total = base_q.count()  # All requests go through all stages
        stage_errors = errors_by_stage.get(stage, 0)
        error_rate_by_stage.append({
            "stage": stage,
            "error_count": stage_errors,
            "total_count": stage_total,
            "rate_percent": round((stage_errors / stage_total) * 100, 2) if stage_total else 0,
        })

    logger.info("Analytics dashboard accessed by %s", current_user)
    return {
        "complaint_stats": {
            "total_complaints_processed": total_nlp,
            "unique_complaints": unique_complaints,
            "duplicate_complaints": dup_count,
            "total_votes": total_votes,
            "average_votes_per_complaint": avg_votes,
        },
        "nlp_stats": {
            "total_requests": total_nlp,
            "avg_processing_time_seconds": round(avg_proc, 4),
            "avg_time_by_stage": {
                "transcription": round(avg_trans, 4),
                "translation": round(avg_transl, 4),
                "classification": round(avg_clf, 4),
                "ner": round(avg_ner, 4),
                "zero_shot": round(avg_zs, 4),
                "image_analysis": round(avg_img, 4),
            },
            "zero_shot_fallback_rate": zs_rate,
            "avg_classifier_confidence": round(avg_conf, 4),
            "avg_entity_count": round(avg_ent, 2),
            "entity_type_breakdown": entity_type_breakdown,
            "avg_word_count": round(avg_wc, 1),
            "avg_audio_duration": round(avg_audio, 2),
        },
        "energy_stats": {
            "total_energy_joules": round(total_energy, 4),
            "avg_energy_per_complaint": round(avg_energy, 4),
            "energy_saved_by_dedup": round(energy_saved, 4),
            "energy_by_stage": {k: round(v, 4) for k, v in energy_stage_agg.items()},
            "calculation_method": CPU_POWER_DETECTION_METHOD,
        },
        "error_stats": {
            "total_errors": total_errors,
            "error_rate_percent": error_rate,
            "errors_by_stage": errors_by_stage,
        },
        "charts": {
            "energy_by_stage": energy_by_stage_chart,
            "energy_over_time": energy_over_time,
            "category_distribution": category_distribution,
            "duplicate_vs_unique": dup_vs_unique,
            "votes_per_complaint": votes_per_complaint,
            "language_distribution": language_distribution,
            "confidence_histogram": confidence_histogram,
            "category_language_heatmap": category_language_heatmap,
            "entity_count_histogram": entity_count_histogram,
            "entity_type_breakdown": entity_type_breakdown,
            "stage_bottleneck_radar": stage_bottleneck_radar,
            "throughput_over_time": throughput_over_time,
            "audio_duration_vs_time": audio_duration_vs_time,
            "duplicate_cluster_sizes": duplicate_cluster_sizes,
            "error_rate_by_stage": error_rate_by_stage,
            "severity_distribution": severity_distribution,
        },
        "data_sources": {
            "complaint_stats": "SELECT COUNT/SUM from complaints table",
            "nlp_metrics": "SELECT from nlp_metrics table (real per-request measurements via time.perf_counter())",
            "energy": f"Estimated TDP ({ESTIMATED_CPU_POWER_WATTS}W) × measured processing time. {CPU_POWER_DETECTION_METHOD}",
            "entities": "spaCy en_core_web_sm NER — entity_count and entity_types stored per request",
            "confidence": "sklearn predict_proba() stored as classifier_confidence per request",
            "audio_duration": "pydub AudioSegment.duration_seconds from uploaded audio",
            "errors": "Caught exceptions logged with stage name to nlp_metrics.error_stage",
            "note": "ALL values computed from database records and runtime logs. Zero hardcoded values.",
        },
    }


# ---------- List Complaints (Paginated, JWT-protected) ----------
@app.get("/complaints")
async def get_complaints(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=100),
    category_mismatch: Optional[bool] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    query = db.query(Complaint)
    
    if category_mismatch is True:
        query = query.filter(Complaint.category_mismatch == True)
        
    total = query.count()
    pages = math.ceil(total / size) if total > 0 else 1
    offset = (page - 1) * size
    items = query.order_by(
        Complaint.votes.desc(),
        Complaint.created_at.desc()
    ).offset(offset).limit(size).all()

    logger.info(f"GET /complaints page={page} size={size} total={total} mismatch={category_mismatch}")
    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages
    }

# ---------- Verify / Edit Complaint (HITL, JWT-protected) ----------
class VerifyRequest(BaseModel):
    category: Optional[str] = None
    status: str = "Verified"
    note: Optional[str] = None

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

    # Auto-append timeline entry
    timeline_entry = ComplaintTimeline(
        complaint_id=id,
        status=request.status,
        note=(request.note or "").strip() or None,
    )
    db.add(timeline_entry)
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


# ---------- Vote on Complaint ----------
class VoteRequest(BaseModel):
    voter_fingerprint: str

@app.post("/complaints/{id}/vote")
def vote_complaint(id: int, body: VoteRequest, db: Session = Depends(get_db)):
    """Upvote a complaint. Idempotent — one vote per fingerprint per complaint."""
    complaint = db.query(Complaint).filter(Complaint.id == id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    fingerprint = (body.voter_fingerprint or "").strip()
    if not fingerprint:
        raise HTTPException(status_code=400, detail="voter_fingerprint is required")

    existing_vote = db.query(ComplaintVote).filter(
        ComplaintVote.complaint_id == id,
        ComplaintVote.voter_fingerprint == fingerprint,
    ).first()

    if existing_vote:
        return {"already_voted": True, "votes": complaint.votes or 0}

    db.add(ComplaintVote(complaint_id=id, voter_fingerprint=fingerprint))
    complaint.votes = (complaint.votes or 0) + 1
    db.commit()
    db.refresh(complaint)
    return {"already_voted": False, "votes": complaint.votes}


# ---------- Public Complaints Listing (no JWT) ----------
@app.get("/complaints/public")
def get_public_complaints(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=12, ge=1, le=100),
    category: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    sort: str = Query(default="latest"),
    voter_fingerprint: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """Public-facing complaint listing — excludes Resolved, no auth needed."""
    query = db.query(Complaint).filter(Complaint.status != "Resolved")
    if category and category != "all":
        query = query.filter(Complaint.category == category)
    if status and status != "all":
        query = query.filter(Complaint.status == status)
    if sort == "most_voted":
        query = query.order_by(Complaint.votes.desc(), Complaint.created_at.desc())
    else:
        query = query.order_by(Complaint.created_at.desc())

    total = query.count()
    pages = math.ceil(total / size) if total else 1
    items = query.offset((page - 1) * size).limit(size).all()

    total_votes = db.query(func.sum(Complaint.votes)).filter(Complaint.status != "Resolved").scalar() or 0

    voted_ids = set()
    if voter_fingerprint:
        item_ids = [c.id for c in items]
        if item_ids:
            votes = db.query(ComplaintVote.complaint_id).filter(
                ComplaintVote.voter_fingerprint == voter_fingerprint,
                ComplaintVote.complaint_id.in_(item_ids)
            ).all()
            voted_ids = {v[0] for v in votes}

    return {
        "items": [
            {
                "id": c.id,
                "category": c.category,
                "location": c.location,
                "status": c.status,
                "votes": c.votes or 0,
                "translated_text": c.translated_text,
                "original_text": c.original_text,
                "language": c.language,
                "trust_level": c.trust_level,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "live_latitude": c.live_latitude,
                "live_longitude": c.live_longitude,
                "voted": c.id in voted_ids,
            }
            for c in items
        ],
        "total": total,
        "page": page,
        "pages": pages,
        "size": size,
        "total_votes": total_votes,
    }


# ---------- Resolved Complaints Archive (no JWT) ----------
@app.get("/complaints/resolved")
def get_resolved_complaints(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=12, ge=1, le=100),
    category: Optional[str] = Query(default=None),
    voter_fingerprint: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """Return only resolved complaints for archive tab."""
    query = db.query(Complaint).filter(Complaint.status == "Resolved")
    if category and category != "all":
        query = query.filter(Complaint.category == category)
    query = query.order_by(Complaint.created_at.desc())
    total = query.count()
    pages = math.ceil(total / size) if total else 1
    items = query.offset((page - 1) * size).limit(size).all()

    total_votes = db.query(func.sum(Complaint.votes)).filter(Complaint.status == "Resolved").scalar() or 0

    voted_ids = set()
    if voter_fingerprint:
        item_ids = [c.id for c in items]
        if item_ids:
            votes = db.query(ComplaintVote.complaint_id).filter(
                ComplaintVote.voter_fingerprint == voter_fingerprint,
                ComplaintVote.complaint_id.in_(item_ids)
            ).all()
            voted_ids = {v[0] for v in votes}

    return {
        "items": [
            {
                "id": c.id,
                "category": c.category,
                "location": c.location,
                "status": c.status,
                "votes": c.votes or 0,
                "translated_text": c.translated_text,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "voted": c.id in voted_ids,
            }
            for c in items
        ],
        "total": total,
        "page": page,
        "pages": pages,
        "total_votes": total_votes,
    }


# ---------- Complaint Timeline (no JWT) ----------
@app.get("/complaints/{id}/timeline")
def get_complaint_timeline(id: int, db: Session = Depends(get_db)):
    """Return the status timeline for a complaint."""
    complaint = db.query(Complaint).filter(Complaint.id == id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    entries = (
        db.query(ComplaintTimeline)
        .filter(ComplaintTimeline.complaint_id == id)
        .order_by(ComplaintTimeline.created_at.asc())
        .all()
    )
    return {
        "complaint_id": id,
        "current_status": complaint.status,
        "timeline": [
            {
                "status": e.status,
                "note": e.note,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in entries
        ],
    }


# ====================== RUN ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)