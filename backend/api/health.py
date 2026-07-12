"""
backend/api/health.py
=====================
Liveness + readiness probes and model status endpoint.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text as sql_text

from backend.database import engine
from backend.services.ai_service import MODEL_RUNTIME_INFO
from backend.config import (
    ENABLE_ZERO_SHOT_FALLBACK,
    ZERO_SHOT_MODEL_NAME,
    ZERO_SHOT_MIN_CONFIDENCE,
    ZERO_SHOT_MIN_SCORE,
    ZERO_SHOT_SPARSE_MIN_SCORE,
    PRIMARY_MIN_EXPLANATORY_FEATURES,
)
import backend.config as _cfg

router = APIRouter(tags=["ops"])


@router.get("/health")
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
                "zero_shot_fallback_loaded": _cfg.zero_shot_classifier is not None,
            },
        },
        status_code=200 if (db_ok and model_ok) else 503,
    )


@router.get("/model/status")
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
            "loaded": _cfg.zero_shot_classifier is not None,
            "model": ZERO_SHOT_MODEL_NAME,
            "min_primary_confidence": ZERO_SHOT_MIN_CONFIDENCE,
            "min_semantic_score": ZERO_SHOT_MIN_SCORE,
            "min_semantic_score_sparse": ZERO_SHOT_SPARSE_MIN_SCORE,
            "primary_min_explanatory_features": PRIMARY_MIN_EXPLANATORY_FEATURES,
        },
    }
