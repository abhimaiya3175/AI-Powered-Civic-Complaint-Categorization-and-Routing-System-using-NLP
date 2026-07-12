"""
backend/services/ai_service.py
================================
ML classifier (TF-IDF + NB), zero-shot fallback, and NLP model loading.
"""

import asyncio
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import spacy
import whisper
from transformers import pipeline

from backend.config import (
    MODEL_PATH_CANDIDATES,
    ENABLE_ZERO_SHOT_FALLBACK,
    ZERO_SHOT_MODEL_NAME,
    ZERO_SHOT_MIN_CONFIDENCE,
    ZERO_SHOT_MIN_SCORE,
    ZERO_SHOT_SPARSE_MIN_SCORE,
    PRIMARY_MIN_EXPLANATORY_FEATURES,
    GENERIC_PRIMARY_FEATURE_TERMS,
    CATEGORY_SEMANTIC_HINTS,
    logger,
    zero_shot_classifier,
    zero_shot_classifier_lock,
)
import backend.config as _cfg  # needed to mutate the module-level singleton

# ── NLP feature helpers (root-level modules) ─────────────────────────
from nlp_features import build_multilingual_classification_text
from image_features import IMAGE_BACKEND, load_yolo_model


# ====================== CLASSIFIER LOADING ======================

def load_classifier_assets() -> Tuple[Dict[str, Any], str]:
    """Load classifier/vectorizer from the first valid model artifact path."""
    seen = set()
    candidates = []
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


# ====================== ZERO-SHOT CLASSIFIER ======================

def get_zero_shot_classifier():
    """Lazily initialize a semantic zero-shot classifier for low-confidence fallback."""
    # Access module-level singleton
    if _cfg.zero_shot_classifier is not None:
        return _cfg.zero_shot_classifier

    with _cfg.zero_shot_classifier_lock:
        if _cfg.zero_shot_classifier is not None:
            return _cfg.zero_shot_classifier

        logger.info("Loading zero-shot semantic classifier: %s", ZERO_SHOT_MODEL_NAME)
        _cfg.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model=ZERO_SHOT_MODEL_NAME,
            device=-1,
        )
        logger.info("Zero-shot semantic classifier loaded successfully")

    return _cfg.zero_shot_classifier


def predict_zero_shot_category_sync(text: str, categories: list) -> Tuple[Optional[str], float]:
    """Run semantic category prediction synchronously (used via asyncio.to_thread)."""
    if not text or not categories:
        return None, 0.0

    clf_zero_shot = get_zero_shot_classifier()

    semantic_to_category: Dict[str, str] = {}
    candidate_labels: list = []
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


# ====================== CATEGORY EXPLANATION ======================

def base_category_explanation(text: str) -> Dict[str, Any]:
    return {
        "method": "tfidf_multinomial_nb",
        "summary": "Top statistically weighted terms from the trained NLP classifier (not rule-based).",
        "classification_text": (text or "").strip(),
        "confidence": None,
        "top_features": [],
        "highlight_terms": [],
    }


def explain_category_prediction(
    text: str,
    text_vector,
    predicted_label: str,
    vectorizer,
    clf,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Explain why a category was predicted by ranking feature contributions
    from the trained TF-IDF + Naive Bayes model."""
    explanation = base_category_explanation(text)

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
            rival_idx = next(
                (idx for idx in ranked_class_indices if idx != predicted_idx), predicted_idx
            )

        feature_names = vectorizer.get_feature_names_out()
        row = text_vector.tocsr()[0]

        if not hasattr(clf, "feature_log_prob_"):
            return explanation

        pred_log_probs = clf.feature_log_prob_[predicted_idx]
        rival_log_probs = (
            clf.feature_log_prob_[rival_idx] if rival_idx != predicted_idx else pred_log_probs
        )

        scored_terms = []
        for feature_idx, tfidf_value in zip(row.indices, row.data):
            term = str(feature_names[feature_idx]).strip()
            if not term:
                continue
            if term.startswith("char_wb__"):
                continue
            if term.startswith("word__"):
                term = term.replace("word__", "", 1)

            weight_delta = float(pred_log_probs[feature_idx] - rival_log_probs[feature_idx])
            contribution = float(tfidf_value) * weight_delta
            if contribution <= 0:
                continue

            scored_terms.append({
                "term": term,
                "contribution": contribution,
                "tfidf": float(tfidf_value),
                "weight_delta": weight_delta,
            })

        if not scored_terms:
            for feature_idx, tfidf_value in zip(row.indices, row.data):
                term = str(feature_names[feature_idx]).strip()
                if not term or term.startswith("char_wb__"):
                    continue
                if term.startswith("word__"):
                    term = term.replace("word__", "", 1)
                scored_terms.append({
                    "term": term,
                    "contribution": float(tfidf_value),
                    "tfidf": float(tfidf_value),
                    "weight_delta": 0.0,
                })

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


# ====================== MODULE-LEVEL MODEL LOADING ======================
# These run once at import time (application startup).

logger.info("Loading model_bbmp.pkl …")
_model_package, _loaded_model_path = load_classifier_assets()
vectorizer = _model_package["vectorizer"]
clf = _model_package["classifier"]

if hasattr(vectorizer, "get_feature_names_out"):
    _vocab_size = len(vectorizer.get_feature_names_out())
else:
    _vocab_size = len(getattr(vectorizer, "vocabulary_", {}))

MODEL_RUNTIME_INFO: Dict[str, Any] = {
    "path": _loaded_model_path,
    "classes": [str(item) for item in getattr(clf, "classes_", [])],
    "class_count": len(getattr(clf, "classes_", [])),
    "vocab_size": _vocab_size,
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

if IMAGE_BACKEND == "yolo":
    logger.info("Loading YOLOv8n-seg pothole detection model …")
    load_yolo_model()
else:
    logger.info("Florence-2 image analysis configured (lazy-load on first request)")
