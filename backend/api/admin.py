"""
backend/api/admin.py
=====================
Admin-only routes: complaint stats and map view.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.dependencies import get_db
from backend.core.security import get_current_user
from backend.models import Complaint
from backend.config import logger

router = APIRouter()


# ---------- Complaint Statistics ----------
@router.get("/complaints/stats")
async def get_stats(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    total = db.query(Complaint).count()
    pending = db.query(Complaint).filter(Complaint.status == "pending").count()
    verified = db.query(Complaint).filter(Complaint.status == "Verified").count()
    total_votes = db.query(func.sum(Complaint.votes)).scalar() or 0

    by_category = (
        db.query(Complaint.category, func.count(Complaint.id).label("count"))
        .group_by(Complaint.category)
        .all()
    )

    by_language = (
        db.query(Complaint.language, func.count(Complaint.id).label("count"))
        .group_by(Complaint.language)
        .all()
    )

    logger.info("Stats endpoint called.")
    return {
        "total": total,
        "pending": pending,
        "verified": verified,
        "total_votes": total_votes,
        "by_category": [{"category": r[0], "count": r[1]} for r in by_category],
        "by_language": [{"language": r[0], "count": r[1]} for r in by_language],
    }


# ---------- Map Complaints (JWT-protected, all with GPS, non-resolved) ----------
@router.get("/complaints/map", tags=["complaints"])
def get_map_complaints(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Return all non-resolved complaints that have GPS coordinates, for map display."""
    items = (
        db.query(Complaint)
        .filter(
            Complaint.status != "Resolved",
            Complaint.live_latitude.isnot(None),
            Complaint.live_longitude.isnot(None),
        )
        .order_by(Complaint.created_at.desc())
        .all()
    )
    return {
        "items": [
            {
                "id": c.id,
                "category": c.category,
                "location": c.location,
                "status": c.status,
                "votes": c.votes or 0,
                "live_latitude": c.live_latitude,
                "live_longitude": c.live_longitude,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in items
        ]
    }
