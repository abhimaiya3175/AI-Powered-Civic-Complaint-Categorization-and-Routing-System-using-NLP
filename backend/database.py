"""
backend/database.py
===================
Database engine, session factory, and declarative base.
Falls back to SQLite when PostgreSQL is unreachable.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import DATABASE_URL, DB_HOST, DB_PORT, DB_NAME, logger

# ── Engine (PostgreSQL → SQLite fallback) ─────────────────────────────
try:
    engine = create_engine(DATABASE_URL)
    engine.connect()
    logger.info("Connected to PostgreSQL at %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
except Exception as e:
    logger.warning("PostgreSQL connection failed — falling back to SQLite. Error: %s", e)
    _sqlite_url = "sqlite:///complaints.db"
    engine = create_engine(_sqlite_url, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
