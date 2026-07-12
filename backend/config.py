"""
backend/config.py
=================
All environment variables, constants, and application-wide configuration.
Loaded once at import time; every other module imports from here.
"""

import os
import re
import sys
import threading
import logging
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

# ── stdout/stderr encoding (Windows cp1252 safety) ────────────────────
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
        logging.FileHandler("bbmp_complaints.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("bbmp")

# ====================== JWT / AUTH ======================
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY is missing. Please set it in your .env file.")

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# ====================== TRANSLATION MODELS ======================
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

# ====================== DATABASE ======================
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "bbmp_complaints")

import urllib.parse
_encoded_password = urllib.parse.quote_plus(DB_PASSWORD or "")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{DB_USER}:{_encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

# ====================== FILE UPLOAD VALIDATION ======================
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

# ====================== GPS / IMAGE VERIFICATION ======================
GPS_TOLERANCE_METERS = 100.0
MAX_IMAGE_AGE_SECONDS = 10 * 60

# ====================== ML MODEL PATHS ======================
MODEL_PATH_CANDIDATES = [
    os.getenv("MODEL_PATH", "").strip(),
    "Models/model_bbmp.pkl",
    "model_bbmp.pkl",
]

# ====================== ZERO-SHOT FALLBACK ======================
ENABLE_ZERO_SHOT_FALLBACK = os.getenv("ENABLE_ZERO_SHOT_FALLBACK", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
ZERO_SHOT_MODEL_NAME = os.getenv("ZERO_SHOT_MODEL_NAME", "valhalla/distilbart-mnli-12-1").strip()
ZERO_SHOT_MIN_CONFIDENCE = float(os.getenv("ZERO_SHOT_MIN_CONFIDENCE", "0.85"))
ZERO_SHOT_MIN_SCORE = float(os.getenv("ZERO_SHOT_MIN_SCORE", "0.55"))
ZERO_SHOT_SPARSE_MIN_SCORE = float(os.getenv("ZERO_SHOT_SPARSE_MIN_SCORE", "0.60"))
PRIMARY_MIN_EXPLANATORY_FEATURES = int(os.getenv("PRIMARY_MIN_EXPLANATORY_FEATURES", "2"))
IMAGE_RECONCILE_CONFIDENCE_THRESHOLD = float(os.getenv("IMAGE_RECONCILE_CONFIDENCE_THRESHOLD", "0.6"))

GENERIC_PRIMARY_FEATURE_TERMS = {
    "street", "road", "area", "near", "public", "issue", "problem", "big", "small",
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

# ====================== KANNADA GLOSSARY ======================
KANNADA_POTHOLE_TERMS = {
    "ಗುಂಡಿ", "ಗುಂಡಿಗಳು", "ಗುಂಡಿಯ", "ಗುಂಡಿಯನ್ನು",
    "ಗುಂಡಿಯಲ್ಲಿ", "ಗುಂಡಿಗಳಿಗೆ", "ಗುಂಡಿಗಳ", "ಗಂಡಿ",
}
KANNADA_POTHOLE_TRANSLIT_PATTERN = re.compile(
    r"\bgund[iy](?:galu|ge|alli|inda|yalli)?\b", re.IGNORECASE
)

# ====================== DUPLICATE DETECTION ======================
DUPLICATE_RADIUS_KM = 0.5           # wide radius when text similarity is also high
DUPLICATE_RADIUS_TIGHT_KM = 0.15   # 150m tight radius for GPS-only match
DUPLICATE_WINDOW_DAYS = 180
DUPLICATE_TEXT_SIMILARITY_THRESHOLD = 0.25  # Jaccard token overlap to confirm duplicate

# ====================== ZERO-SHOT GLOBAL STATE ======================
# Kept here so all modules share a single instance
zero_shot_classifier = None
zero_shot_classifier_lock = threading.Lock()
