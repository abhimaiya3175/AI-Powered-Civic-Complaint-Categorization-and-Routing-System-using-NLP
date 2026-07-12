"""
backend/core/dependencies.py
=============================
FastAPI dependency functions (database session, etc.).
Kept separate from security.py to break potential circular imports.
"""

from backend.database import SessionLocal


def get_db():
    """Yield a DB session and ensure it is closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
