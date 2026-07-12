"""
backend/main.py
================
FastAPI application factory for the refactored backend package.

Run with:
    uvicorn backend.main:app --reload

The original root main.py is intentionally left untouched for backward compatibility.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Trigger model and DB setup at import time ─────────────────────────
# Importing models ensures Base.metadata.create_all() and schema upgrades run.
import backend.models  # noqa: F401
# Importing ai_service loads pkl, spaCy, Whisper, and (if yolo) YOLO model.
import backend.services.ai_service  # noqa: F401

from backend.core.startup import preload_translation_models

# ── API routers ───────────────────────────────────────────────────────
from backend.api import health, auth, complaints, admin, analytics

app = FastAPI(title="Multilingual Civic Complaint System (BBMP)")

# ── CORS ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup event ─────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    """Preload IndicTrans2 / NLLB translation models into app.state."""
    await preload_translation_models(app.state)

# Route registration order matters: literal routes (stats, map, public, resolved)
# must be registered before parameterised routes ({id}) so FastAPI prefers them.
# admin.router has /complaints/stats and /complaints/map — register it first.
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(admin.router)       # /complaints/stats, /complaints/map
app.include_router(complaints.router)  # /complaints/{id}/..., /complaints/public, etc.
app.include_router(analytics.router)


# ── Local dev entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
