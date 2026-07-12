"""
backend/api/analytics.py
=========================
NLP analytics dashboard endpoint (JWT-protected).
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.core.dependencies import get_db
from backend.core.security import get_current_user
from backend.services.analytics_service import build_analytics_dashboard
from backend.config import logger

router = APIRouter(tags=["analytics"])


@router.get("/analytics/dashboard")
async def analytics_dashboard(
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    language: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Comprehensive NLP analytics dashboard. ALL values from real DB data."""
    logger.info("Analytics dashboard accessed by %s", current_user)
    return build_analytics_dashboard(db, start_date=start_date, end_date=end_date, language=language)
