"""
backend/models.py
=================
SQLAlchemy ORM models — AdminUser, Complaint, ComplaintVote,
ComplaintTimeline, NlpMetric.

Schema-upgrade helpers are also here so database.py stays clean.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, func, inspect,
    text as sql_text,
)

from backend.database import Base, engine


# ====================== MODELS ======================

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
    # Image detection (YOLOv8n-seg pothole/road-damage detection — backward compat)
    detected_objects = Column(Text)         # JSON: [{class, confidence, bbox, severity}]
    annotated_image_path = Column(String)   # Path to annotated image (nullable)
    pothole_severity = Column(String)       # Overall severity: Clear/Low/Medium/High/Severe
    image_suggested_category = Column(String)  # Category suggested by image analysis, nullable
    category_mismatch = Column(Boolean, default=False)  # True if image analysis strongly disagrees with NLP
    # Florence-2 Image Analysis
    florence_status = Column(String, default=None)          # processing|success|unavailable|timeout|error|None
    florence_caption = Column(Text)                         # Short image caption
    florence_damaged_object = Column(String)                # Top detected object from <OD>
    florence_problem_type = Column(String)                  # Matched problem type keyword
    florence_severity = Column(String)                      # Keyword-based severity from caption
    florence_evidence = Column(Text)                        # Raw <MORE_DETAILED_CAPTION> text
    florence_processing_time = Column(Float)                # Seconds
    florence_all_categories = Column(Text)                  # JSON: [{category, matched_keywords, score}]
    # Cross-Modal Verification
    cross_modal_result = Column(String)                    # match|mismatch|image_unclear|None
    cross_modal_nlp_category = Column(String)              # NLP side
    cross_modal_image_category = Column(String)            # Image side
    manual_review_required = Column(Boolean, default=False)


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


# ====================== SCHEMA UPGRADE HELPERS ======================

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


def ensure_complaints_schema_upgrades() -> None:
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
        # Florence-2 Image Analysis
        "florence_status": "VARCHAR",
        "florence_caption": "TEXT",
        "florence_damaged_object": "VARCHAR",
        "florence_problem_type": "VARCHAR",
        "florence_severity": "VARCHAR",
        "florence_evidence": "TEXT",
        "florence_processing_time": "FLOAT",
        "florence_all_categories": "TEXT",
        # Cross-Modal Verification
        "cross_modal_result": "VARCHAR",
        "cross_modal_nlp_category": "VARCHAR",
        "cross_modal_image_category": "VARCHAR",
        "manual_review_required": "BOOLEAN DEFAULT FALSE",
    })


def ensure_nlp_metrics_schema_upgrades() -> None:
    """Add image analysis columns to nlp_metrics for existing databases."""
    _ensure_table_columns("nlp_metrics", {
        "image_analysis_time": "FLOAT DEFAULT 0.0",
        "detected_object_count": "INTEGER DEFAULT 0",
        "image_model_confidence": "FLOAT",
        "pothole_severity": "VARCHAR",
    })


# Create all tables and run schema upgrades once at import time.
Base.metadata.create_all(bind=engine)
ensure_complaints_schema_upgrades()
ensure_nlp_metrics_schema_upgrades()
