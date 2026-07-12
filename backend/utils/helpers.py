"""
backend/utils/helpers.py
========================
Miscellaneous request-validation and text helpers used across multiple layers.
"""

import re
from datetime import datetime, timezone

from fastapi import HTTPException, UploadFile

from backend.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_AUDIO_TYPES,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_IMAGE_TYPES,
    SUPPORTED_LANGUAGES,
    KANNADA_POTHOLE_TERMS,
    KANNADA_POTHOLE_TRANSLIT_PATTERN,
    logger,
)


# ── spaCy NER location extraction ────────────────────────────────────

def extract_location(text: str, nlp_model) -> str:
    """Extract geographic entities from text using spaCy NER."""
    doc = nlp_model(text)
    locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
    return ", ".join(locations) if locations else "Unknown"


# ── File validation ───────────────────────────────────────────────────

def validate_audio_file(file: UploadFile) -> None:
    """Reject uploads that are not audio files (by extension + MIME type)."""
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()
    content_type = (file.content_type or "").lower()

    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        logger.warning("Rejected file upload: %s — not an audio file", file.filename)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )
    if content_type and content_type not in ALLOWED_AUDIO_TYPES:
        logger.warning("Rejected file upload: %s — not an audio file", file.filename)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio content type '{content_type}'.",
        )


def validate_image_file(file: UploadFile) -> None:
    """Reject uploads that are not accepted image files."""
    import os
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


# ── Request field parsing ─────────────────────────────────────────────

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


# ── Kannada civic translation glossary ───────────────────────────────

# Maps (source_term_in_kn/hi, translated_civic_phrase)
# Used to patch garbled IndicTrans2 outputs before classification.
_CIVIC_GLOSSARY_KN = [
    # Street Light
    (r"ಬೀದಿ\s*ದೀಪ|ಬೀದಿದೀಪ",              "street light not working"),
    (r"ವಿದ್ಯುತ್\s*ದೀಪ|ವಿದ್ಯುತ್ದೀಪ",       "electrical street light broken"),
    (r"ದೀಪ\s*ಹಾಳ|ದೀಪ\s*ಕೆಟ್ಟ",             "street light damaged"),
    # Road Repair (pothole — already handled, but kept for completeness)
    (r"ಗುಂಡಿ(?:ಗಳ|ಗಳಿವೆ|ಗೆ)?",            "road pothole"),
    (r"ರಸ್ತೆ\s*(?:ಹಾಳ|ರಿಪೇರಿ|ಮರಮ್ಮತ್)",   "road repair needed"),
    # Drainage / SWD
    (r"ಚರಂಡಿ\s*(?:ಬ್ಲಾಕ್|ಕಟ್ಟ|ತುಂಬ)",      "drain blocked overflow"),
    (r"ನೀರು\s*ನಿಂತ|ನೀರು\s*ತುಂಬ",          "waterlogging stagnant water"),
    # Garbage
    (r"ಕಸ\s*(?:ತೆಗೆ|ಸಂಗ್ರಹ|ಎತ್ತ)",          "garbage not collected"),
    (r"ತ್ಯಾಜ್ಯ\s*(?:ರಾಶಿ|ತುಂಬ)",           "waste pile garbage"),
    # Water Supply
    (r"ನೀರು\s*(?:ಬರ|ಸರಬರ|ಸಪ್ಲೈ)\s*ಇಲ್ಲ",  "no water supply"),
    (r"ನಲ್ಲಿ\s*ನೀರ",                       "tap water supply problem"),
    # Health / Sanitation
    (r"ಸೊಳ್ಳೆ\s*(?:ಸಮಸ್ಯ|ಉಪದ್ರ)",          "mosquito health problem"),
    (r"ಡೆಂಗ್ಯೂ|ಮಲೇರಿಯ",                   "dengue malaria health"),
    # Parks
    (r"ಉದ್ಯಾನ\s*(?:ಹಾಳ|ಸರಿ\s*ಇಲ್ಲ)",      "park damaged maintenance"),
    (r"ಆಟ\s*ಸಾಮಗ್ರಿ\s*(?:ಮುರಿ|ಹಾಳ)",     "playground equipment broken"),
    # Traffic
    (r"ಟ್ರಾಫಿಕ್\s*(?:ಸಿಗ್ನಲ್|ಜಾಮ್)",     "traffic signal jam"),
    # Town Planning
    (r"ಅನಧಿಕೃತ\s*ಕಟ್ಟಡ",                  "unauthorized building construction"),
    # Veterinary
    (r"ಬೀದಿ\s*ನಾಯಿ|ತಿರುಕ\s*ನಾಯಿ",        "stray dog veterinary"),
    # Advertisement
    (r"ಅಕ್ರಮ\s*ಜಾಹೀರಾತ",                  "illegal advertisement hoarding"),
]

_CIVIC_GLOSSARY_HI = [
    # Street Light
    (r"बिजली\s*(?:बत्ती|लाइट|खंभ)",       "street light not working"),
    (r"सड़क\s*बत्ती|स्ट्रीट\s*लाइट",       "street light broken"),
    # Road Repair
    (r"गड्ढ[ेों]",                         "road pothole"),
    (r"सड़क\s*(?:टूट|मरम्मत|खराब)",        "road repair broken"),
    # Drainage / SWD
    (r"नाली\s*(?:बंद|जाम|भर)",             "drain blocked overflow"),
    (r"जलभराव|पानी\s*(?:भर|जम)",           "waterlogging flooding"),
    # Garbage
    (r"कचरा\s*(?:नहीं\s*उठा|जमा|ढेर)",    "garbage not collected"),
    # Water Supply
    (r"पानी\s*(?:नहीं|बंद|सप्लाई)",        "water supply problem"),
    (r"नल\s*(?:में\s*पानी\s*नहीं|बंद)",    "tap water no supply"),
    # Health / Sanitation
    (r"मच्छर\s*(?:समस्या|ज्यादा)",         "mosquito health problem"),
    (r"डेंगू|मलेरिया",                     "dengue malaria health"),
    # Parks
    (r"पार्क\s*(?:खराब|टूट|झूल)",         "park damaged equipment broken"),
    # Traffic
    (r"ट्रैफिक\s*(?:सिग्नल|जाम)",          "traffic signal jam"),
    # Veterinary
    (r"आवारा\s*कुत्त",                     "stray dog veterinary"),
    # Revenue
    (r"खाता\s*(?:ट्रांसफर|वर्गाव)",       "khata transfer revenue"),
]


def _is_garbled_translation(source_text: str, translated_text: str) -> bool:
    """Heuristic: detect suspiciously garbled/irrelevant translations.

    Returns True when the translation is likely garbage (e.g., "It's a good day"
    for a Kannada street light complaint). Checks:
    1. Translation is very short (< 4 words) vs. longer source.
    2. Translation contains no civic indicator words.
    """
    src_words = len((source_text or "").split())
    tgt_words = len((translated_text or "").split())
    if src_words >= 4 and tgt_words <= 3:
        return True

    civic_indicators = {
        "road", "pothole", "light", "street", "drain", "water", "garbage",
        "waste", "traffic", "park", "repair", "supply", "health", "mosquito",
        "building", "construction", "stray", "dog", "hoarding", "khata",
        "complaint", "issue", "problem", "not working", "broken", "damaged",
    }
    tgt_lower = (translated_text or "").lower()
    has_civic = any(ind in tgt_lower for ind in civic_indicators)
    return not has_civic


def apply_civic_translation_glossary(
    source_text: str,
    translated_text: str,
    source_language: str,
    target_language: str,
) -> str:
    """Patch known civic-term mistranslations for Kannada/Hindi → English outputs.

    Now covers all 14 BBMP complaint categories, not just potholes.
    When a garbled translation is detected, the function rebuilds the English
    text from civic source terms found in the original Kannada/Hindi text.
    """
    candidate = (translated_text or "").strip()
    source_clean = (source_text or "").strip()

    if not candidate or target_language != "en":
        return candidate

    is_kn = source_language == "kn" or bool(re.search(r"[\u0C80-\u0CFF]", source_clean))
    is_hi = source_language == "hi" or bool(re.search(r"[\u0900-\u097F]", source_clean))

    if not (is_kn or is_hi):
        return candidate

    glossary = _CIVIC_GLOSSARY_KN if is_kn else _CIVIC_GLOSSARY_HI
    matched_phrases = []

    for pattern, civic_phrase in glossary:
        if re.search(pattern, source_clean, flags=re.IGNORECASE):
            matched_phrases.append(civic_phrase)

    if not matched_phrases:
        # No civic glossary terms found — legacy pothole-only patch still runs
        return _apply_pothole_patch(source_clean, candidate, is_kn)

    # If translation is garbled AND we found civic source terms, rebuild from glossary
    if _is_garbled_translation(source_clean, candidate):
        rebuilt = ". ".join(phrase.capitalize() for phrase in matched_phrases) + "."
        logger.info(
            "Civic glossary rebuilt garbled translation: '%s' -> '%s' (source: %s)",
            candidate, rebuilt, source_clean[:60],
        )
        return rebuilt

    # Translation looks OK — still inject pothole correction if needed
    return _apply_pothole_patch(source_clean, candidate, is_kn)


def _apply_pothole_patch(source_clean: str, candidate: str, is_kn: bool) -> str:
    """Legacy pothole-specific patch — preserved from original implementation."""
    from backend.config import KANNADA_POTHOLE_TERMS, KANNADA_POTHOLE_TRANSLIT_PATTERN

    if not is_kn:
        return candidate

    translit_match = bool(KANNADA_POTHOLE_TRANSLIT_PATTERN.search(source_clean.lower()))
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
