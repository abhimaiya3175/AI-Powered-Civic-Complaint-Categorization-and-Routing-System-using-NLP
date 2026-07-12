"""
backend/services/complaint_service.py
=======================================
Complaint submission pipeline: NLP processing, trust-level assignment,
duplicate detection, and database persistence.
"""

import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.config import (
    ENABLE_ZERO_SHOT_FALLBACK,
    GPS_TOLERANCE_METERS,
    MAX_IMAGE_AGE_SECONDS,
    ZERO_SHOT_MIN_CONFIDENCE,
    ZERO_SHOT_MIN_SCORE,
    ZERO_SHOT_SPARSE_MIN_SCORE,
    GENERIC_PRIMARY_FEATURE_TERMS,
    SUPPORTED_LANGUAGES,
    CATEGORY_SEMANTIC_HINTS,
    logger,
)
from backend.models import Complaint, ComplaintTimeline, ComplaintVote, NlpMetric
from backend.services.audio_service import transcribe_audio
from backend.services.translation_service import translate_text_with_indictrans2
from backend.services.ai_service import (
    vectorizer, clf, nlp,
    whisper_model,
    MODEL_RUNTIME_INFO,
    base_category_explanation,
    explain_category_prediction,
    predict_zero_shot_category_sync,
)
from backend.utils.helpers import (
    validate_audio_file,
    validate_image_file,
    parse_client_timestamp,
    normalize_language_code,
)
from backend.utils.gps import haversine_distance_meters
from backend.utils.exif import extract_exif_location_and_time
from backend.utils.duplicate import find_duplicate_complaint
from backend.utils.energy import detect_cpu_power_watts
from nlp_features import build_multilingual_classification_text

# Detect CPU power once at module load
ESTIMATED_CPU_POWER_WATTS, CPU_POWER_DETECTION_METHOD = detect_cpu_power_watts()
logger.info(
    "CPU power estimation: %.1fW — %s",
    ESTIMATED_CPU_POWER_WATTS,
    CPU_POWER_DETECTION_METHOD,
)


async def submit_complaint(
    file: Optional[UploadFile],
    image: Optional[UploadFile],
    live_latitude: float,
    live_longitude: float,
    live_location_timestamp: str,
    text_note: str,
    language: str,
    target_language: str,
    voter_fingerprint: str,
    db: Session,
    app_state,
) -> Dict[str, Any]:
    """
    Full NLP pipeline: receive audio → transcribe (Whisper/Google STT) →
    translate (IndicTrans2) → classify → save to DB.
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
    _nlp_image_analysis_time = 0.0
    _nlp_detected_object_count = 0
    _nlp_image_model_confidence = None
    _nlp_pothole_severity = None
    _nlp_detected_objects = None

    # 0. Validate live location fields (mandatory)
    if not (-90 <= live_latitude <= 90) or not (-180 <= live_longitude <= 180):
        raise HTTPException(status_code=400, detail="Live location coordinates are invalid.")

    live_location_at = parse_client_timestamp(live_location_timestamp, "live_location_timestamp")

    submitted_text = (text_note or "").strip()
    recording_language = normalize_language_code(language, "language")
    target_language_code = normalize_language_code(target_language, "target_language")

    if file is None and not submitted_text and image is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either audio, complaint text, or an image along with live location.",
        )

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
                detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB.",
            )
        await file.seek(0)

    if image is not None:
        image_contents = await image.read()
        if len(image_contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB.",
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
        logger.info("Received audio file: %s", file.filename)

    if image is not None:
        image_id = str(uuid.uuid4())
        image_ext = os.path.splitext(image.filename or ".jpg")[1] or ".jpg"
        image_path = f"uploads/{image_id}{image_ext}"
        with open(image_path, "wb") as f:
            f.write(await image.read())
        logger.info("Received image evidence: %s", image.filename)

    # ── Stage 1: Transcription (timed) ────────────────────────────
    transcribed_text = submitted_text
    detected_language = recording_language

    if submitted_text and not audio_path:
        try:
            import langid
            guessed_lang, _ = langid.classify(submitted_text)
            if guessed_lang in SUPPORTED_LANGUAGES:
                detected_language = guessed_lang
                logger.info(
                    "Auto-detected text language as '%s' (UI passed '%s')",
                    guessed_lang, recording_language,
                )
        except ImportError:
            pass
        except Exception as e:
            logger.warning("langid failed: %s", e)

    _t_transcription_start = time.perf_counter()
    if audio_path:
        (
            transcribed_text,
            detected_language,
            _nlp_audio_duration,
            _t_error_stage,
            _t_error_msg,
        ) = await transcribe_audio(audio_path, recording_language, submitted_text, whisper_model)
        if _t_error_stage:
            _nlp_error_stage = _t_error_stage
            _nlp_error_message = _t_error_msg
    _nlp_transcription_time = time.perf_counter() - _t_transcription_start

    if not transcribed_text and not image_path:
        raise HTTPException(status_code=400, detail="Complaint text or image is required.")

    # ── Stage 2: Translation (timed) ──────────────────────────────
    _t_translation_start = time.perf_counter()
    try:
        translated_text = await translate_text_with_indictrans2(
            transcribed_text, detected_language, target_language_code, app_state
        )
        logger.info(
            "Translated text (%s->%s): %s",
            detected_language, target_language_code, translated_text,
        )

        english_text_for_classification = translated_text
        if target_language_code != "en":
            if detected_language == "en":
                english_text_for_classification = transcribed_text
            else:
                english_text_for_classification = (
                    await translate_text_with_indictrans2(
                        transcribed_text, detected_language, "en", app_state
                    ) or transcribed_text
                )
    except HTTPException:
        raise
    except Exception as e:
        _nlp_error_stage = "translation"
        _nlp_error_message = str(e)
        logger.error("Translation failed: %s", e)
        translated_text = transcribed_text
        english_text_for_classification = transcribed_text
    _nlp_translation_time = time.perf_counter() - _t_translation_start

    # ── Stage 3: Classification (timed) ───────────────────────────
    _t_classification_start = time.perf_counter()
    classification_text = build_multilingual_classification_text(
        transcribed_text, english_text_for_classification
    )
    category_explanation = base_category_explanation(classification_text)
    primary_confidence = None
    try:
        text_vector = vectorizer.transform([classification_text])
        category = str(clf.predict(text_vector)[0])
        primary_category = category
        category_explanation = explain_category_prediction(
            classification_text, text_vector, category, vectorizer, clf
        )

        primary_confidence = category_explanation.get("confidence")
        _nlp_classifier_confidence = (
            float(primary_confidence) if isinstance(primary_confidence, (float, int)) else None
        )
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
        logger.error("Classification failed: %s", e)
        category = "Others"
        primary_category = category
        low_primary_confidence = False
        sparse_primary_signal = False
        generic_primary_signal = False
    _nlp_classification_time = time.perf_counter() - _t_classification_start

    # ── Keyword Override (Priority) ───────────────────────────────────
    # Deterministic overrides for unambiguous keywords. Each override bypasses
    # the ML classifier to guarantee correct routing for clear-cut cases.
    # Checked in specificity order (most specific first).
    _KEYWORD_OVERRIDES = [
        # (regex_pattern, category, trigger_term_label)
        (r'\bpotholes?\b', "Road Repair", "pothole"),
        (r'\broad\s+(?:repair|damage|broken|crack)', "Road Repair", "road damage"),
        (r'\bगड्ढ[ेों]?\b', "Road Repair", "गड्ढा (pothole)"),
        (r'\bರಸ್ತೆ\s*(?:ಗುಂಡಿ|ರಿಪೇರಿ|ಹಾಳ)', "Road Repair", "ರಸ್ತೆ ಗುಂಡಿ (road pothole)"),
        (r'\b(?:waterlogging|water\s*logging|flooded?\s*road|water\s*fill(?:s|ed)\s*(?:the\s*)?road)\b', "Drainage / SWD", "waterlogging"),
        (r'\bcharan[dḍ]i\b', "Drainage / SWD", "charandi (drain)"),
        (r'\bचरंडि|ನೀರು\s*ನಿಂತ|ಚರಂಡಿ\b', "Drainage / SWD", "drain"),
        (r'\b(?:garbage\s+not\s+collected|waste\s+pile|kachra|कचरा|ಕಸ\s*ಸಂಗ್ರಹ)\b', "Garbage / Sanitation", "garbage"),
        (r'\b(?:street\s*light|lamp\s*post)\s+(?:not\s+working|broken|damaged)\b', "Street Light", "street light broken"),
        (r'\bबिजली\s*(?:बत्ती|लाइट)\b', "Street Light", "बिजली बत्ती (street light)"),
        (r'\bವಿದ್ಯುತ್\s*ದೀಪ|ಬೀದಿ\s*ದೀಪ\b', "Street Light", "ಬೀದಿ ದೀಪ (street light)"),
    ]

    for pattern, override_category, trigger_label in _KEYWORD_OVERRIDES:
        if re.search(pattern, english_text_for_classification, flags=re.IGNORECASE) or \
           re.search(pattern, transcribed_text or "", flags=re.IGNORECASE):
            category = override_category
            primary_category = override_category
            primary_confidence = 1.0
            low_primary_confidence = False
            sparse_primary_signal = False
            generic_primary_signal = False
            _nlp_classifier_confidence = 1.0
            category_explanation = {
                "method": "keyword_override",
                "summary": f"Complaint explicitly matches '{trigger_label}', bypassing ML classification to guarantee {override_category} routing.",
                "classification_text": english_text_for_classification,
                "confidence": 1.0,
                "top_features": [{"term": trigger_label, "contribution": 1.0, "tfidf": 1.0, "weight_delta": 1.0, "importance_percent": 100.0}],
                "highlight_terms": [trigger_label],
            }
            break  # First match wins (most specific patterns are listed first)


    # ── Stage 4: Zero-shot fallback (timed) ───────────────────────
    _t_zero_shot_start = time.perf_counter()
    if (
        ENABLE_ZERO_SHOT_FALLBACK
        and english_text_for_classification.strip()
        and (low_primary_confidence or sparse_primary_signal or generic_primary_signal or category == "Others")
    ):
        _nlp_zero_shot_triggered = True
        try:
            candidate_categories = [
                c for c in MODEL_RUNTIME_INFO.get("classes", [])
                if c not in ("Non-Civic",)
            ]
            semantic_category, semantic_score = await asyncio.to_thread(
                predict_zero_shot_category_sync,
                english_text_for_classification,
                candidate_categories,
            )
            _nlp_zero_shot_confidence = float(semantic_score) if semantic_score else 0.0

            semantic_threshold = (
                0.25 if category == "Others" else
                (ZERO_SHOT_MIN_SCORE if low_primary_confidence else ZERO_SHOT_SPARSE_MIN_SCORE)
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

                # Build highlight_terms from the text words that overlap with
                # the winning category's semantic hint phrase
                hint_phrase = CATEGORY_SEMANTIC_HINTS.get(semantic_category, semantic_category)
                hint_words = set(hint_phrase.lower().split())
                text_words = english_text_for_classification.lower().split()
                _zs_highlight = [
                    w for w in text_words
                    if w in hint_words and len(w) > 3
                ][:5]
                if not _zs_highlight:
                    # Fall back to longest words in the text as highlights
                    _zs_highlight = sorted(
                        [w for w in text_words if len(w) > 4],
                        key=len, reverse=True
                    )[:3]

                category_explanation = {
                    "method": "zero_shot_nli_fallback",
                    "summary": f"Primary TF-IDF evidence was low-confidence or sparse, so semantic NLI classification was used (NLP model, not rule-based).",
                    "classification_text": english_text_for_classification,
                    "confidence": round(float(semantic_score), 4),
                    "top_features": [
                        {"term": w, "contribution": round(semantic_score / len(_zs_highlight), 4) if _zs_highlight else 0,
                         "tfidf": 0.0, "weight_delta": 0.0,
                         "importance_percent": round(100.0 / len(_zs_highlight), 1) if _zs_highlight else 0}
                        for w in _zs_highlight
                    ],
                    "highlight_terms": _zs_highlight,
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

    # ── Translation quality assessment ────────────────────────────
    _translation_quality = "good"
    if detected_language != "en" and transcribed_text:
        from backend.utils.helpers import _is_garbled_translation
        if _is_garbled_translation(transcribed_text, english_text_for_classification):
            _translation_quality = "low"
        elif category_explanation.get("method") == "keyword_override" and \
             english_text_for_classification != transcribed_text:
            _translation_quality = "glossary_corrected"

    # ── Decision path string ──────────────────────────────────────
    method = category_explanation.get("method", "")
    conf = category_explanation.get("confidence")
    _top_terms = category_explanation.get("highlight_terms") or []
    if method == "keyword_override":
        _trigger = (_top_terms[0] if _top_terms else "unknown")
        _decision_path = f"keyword_override:{_trigger}"
    elif method == "zero_shot_nli_fallback":
        _decision_path = f"zero_shot_nli:{round(float(conf), 2) if conf else '?'}"
    else:
        _terms_str = ",".join(_top_terms[:3]) if _top_terms else "no_features"
        _decision_path = f"tfidf_nb:{_terms_str}({round(float(conf)*100, 0) if conf else '?'}%)"

    location = f"{live_latitude:.6f}, {live_longitude:.6f}"
    logger.info("Category: %s | Location: %s", category, location)

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
            try:
                os.remove(audio_path)
            except Exception:
                pass
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass
        logger.info("Discarded irrelevant complaint: %s", transcribed_text)
        raise HTTPException(
            status_code=400,
            detail="The submitted audio/text is irrelevant and does not appear to be a valid civic complaint. It has been discarded.",
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
        try:
            image_exif_latitude, image_exif_longitude, image_exif_timestamp = (
                extract_exif_location_and_time(image_path)
            )
            image_live_distance_meters = haversine_distance_meters(
                live_latitude, live_longitude,
                image_exif_latitude, image_exif_longitude,
            )
            if image_live_distance_meters <= GPS_TOLERANCE_METERS:
                image_age_seconds = (datetime.utcnow() - image_exif_timestamp).total_seconds()
                if image_age_seconds <= MAX_IMAGE_AGE_SECONDS:
                    trust_level = "high"
                    verification_mode = "auto_verified"
                    status = "Verified"
                else:
                    logger.info(
                        "Image EXIF timestamp is older than %d seconds, skipping auto-verify",
                        MAX_IMAGE_AGE_SECONDS,
                    )
            else:
                logger.info(
                    "Image GPS distance %.1fm exceeds tolerance, skipping auto-verify",
                    image_live_distance_meters,
                )
        except HTTPException:
            logger.info("Image has no valid EXIF/GPS data — accepting without auto-verification")
        except Exception as exc:
            logger.warning("EXIF extraction failed: %s — accepting image without auto-verification", exc)

    # ── Stage 6: Image analysis — background processing ──────────
    _t_image_analysis_start = time.perf_counter()
    image_suggested_category = None
    category_mismatch = False
    florence_status = "processing" if image_path else None
    _nlp_image_analysis_time = time.perf_counter() - _t_image_analysis_start

    # 8. Duplicate detection
    existing_dup = find_duplicate_complaint(
        db, category, live_latitude, live_longitude, location,
        complaint_text=transcribed_text or english_text_for_classification,
    )

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

    def _build_nlp_metric(**extra) -> NlpMetric:
        return NlpMetric(
            is_duplicate=extra.get("is_duplicate", False),
            complaint_id=extra["complaint_id"],
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
        )

    if existing_dup:
        # Record NLP metric for duplicate
        try:
            db.add(_build_nlp_metric(complaint_id=existing_dup.id, is_duplicate=True))
            db.commit()
        except Exception as metric_exc:
            logger.warning("Failed to record NLP metric for duplicate: %s", metric_exc)

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

        # Clean up uploaded files
        for path in [audio_path, image_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.error("Failed to delete file %s: %s", path, e)

        logger.info(
            "Duplicate detected: new complaint matches existing #%d, votes count updated to %d",
            existing_dup.id, existing_dup.votes,
        )
        return {
            "duplicate": True,
            "id": existing_dup.id,
            "category": existing_dup.category,
            "location": existing_dup.location,
            "status": existing_dup.status,
            "votes": existing_dup.votes or 0,
            "voted": voted,
            "already_reported_by_you": not voted and bool(fp),
            "translation_quality": _translation_quality,
            "decision_path": _decision_path,
            "message": (
                "Your vote has been added to this existing complaint."
                if voted else
                (
                    f"You already reported this issue (#{existing_dup.id}). "
                    f"Current status: {existing_dup.status}. "
                    f"It has {existing_dup.votes or 0} vote(s) so far."
                    if fp else
                    "This issue already exists and has been reported."
                )
            ),
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
        florence_status=florence_status,
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)

    # Spawn background image analysis
    if image_path:
        from backend.services.image_service import run_image_analysis_background
        asyncio.create_task(run_image_analysis_background(complaint.id, image_path, category))

    # Record submitter fingerprint as first vote
    if fp:
        db.add(ComplaintVote(complaint_id=complaint.id, voter_fingerprint=fp))
        db.commit()

    # Auto-create first timeline entry
    db.add(ComplaintTimeline(complaint_id=complaint.id, status="Reported", note="Complaint submitted"))
    db.commit()
    logger.info("Complaint saved with ID: %d, initial votes: %d", complaint.id, complaint.votes)

    # Record NLP metric for new complaint
    try:
        db.add(_build_nlp_metric(complaint_id=complaint.id, is_duplicate=False))
        db.commit()
    except Exception as metric_exc:
        logger.warning("Failed to record NLP metric: %s", metric_exc)

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
        "translation_quality": _translation_quality,
        "decision_path": _decision_path,
        "florence_analysis": {
            "status": complaint.florence_status,
            "caption": complaint.florence_caption,
            "damaged_object": complaint.florence_damaged_object,
            "problem_type": complaint.florence_problem_type,
            "severity": complaint.florence_severity,
            "supporting_evidence": complaint.florence_evidence,
            "processing_time": complaint.florence_processing_time,
            "all_suggested_categories": (
                json.loads(complaint.florence_all_categories)
                if complaint.florence_all_categories else []
            ),
        },
        "cross_modal": {
            "nlp_category": complaint.cross_modal_nlp_category or complaint.category,
            "image_category": complaint.cross_modal_image_category,
            "verification_result": complaint.cross_modal_result,
            "trust_level": complaint.trust_level,
            "manual_review_required": complaint.manual_review_required,
        },
    }
