"""
backend/utils/duplicate.py
===========================
Duplicate-complaint detection query helper.

Uses a tiered strategy:
- Tight GPS radius (150m) alone is enough to flag a duplicate.
- Wide GPS radius (500m) requires text similarity (Jaccard ≥ 0.25) to confirm.
This prevents two distinct complaints in the same area from being merged,
while still catching the same issue reported multiple times from slightly
different vantage points.
"""

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from backend.config import (
    DUPLICATE_RADIUS_KM,
    DUPLICATE_RADIUS_TIGHT_KM,
    DUPLICATE_WINDOW_DAYS,
    DUPLICATE_TEXT_SIMILARITY_THRESHOLD,
)
from backend.utils.gps import haversine_distance_meters


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two strings.

    Returns a value in [0, 1] where 1 = identical token sets.
    Ignores very short words (≤ 2 chars) to avoid stop-word noise.
    """
    tokens_a = {w.lower() for w in (text_a or "").split() if len(w) > 2}
    tokens_b = {w.lower() for w in (text_b or "").split() if len(w) > 2}
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def find_duplicate_complaint(
    db: Session,
    category: str,
    live_latitude: float,
    live_longitude: float,
    location: str,
    complaint_text: str = "",
):
    """Return the first existing open complaint that matches as a duplicate.

    Matching rules (both must apply):
    - Same BBMP category
    - Within the time window (DUPLICATE_WINDOW_DAYS)
    - Either:
        (a) Tight GPS match: distance ≤ DUPLICATE_RADIUS_TIGHT_KM (150m) — no text check needed
        (b) Wide GPS match: distance ≤ DUPLICATE_RADIUS_KM (500m) AND
            Jaccard text similarity ≥ DUPLICATE_TEXT_SIMILARITY_THRESHOLD (0.25)

    Returns None if no duplicate is found.
    """
    from backend.models import Complaint  # local import avoids circular deps

    window_start = datetime.utcnow() - timedelta(days=DUPLICATE_WINDOW_DAYS)
    dup_query = db.query(Complaint).filter(
        Complaint.category == category,
        Complaint.status != "Resolved",
        Complaint.created_at >= window_start,
    )

    for candidate in dup_query.all():
        if candidate.live_latitude and candidate.live_longitude:
            dist = haversine_distance_meters(
                live_latitude, live_longitude,
                candidate.live_latitude, candidate.live_longitude,
            )
            tight_radius_m = DUPLICATE_RADIUS_TIGHT_KM * 1000  # 150m
            wide_radius_m = DUPLICATE_RADIUS_KM * 1000          # 500m

            if dist <= tight_radius_m:
                # Close enough — no need for text check
                return candidate

            if dist <= wide_radius_m:
                # Borderline distance — require text similarity to confirm
                sim = _jaccard_similarity(
                    complaint_text,
                    candidate.original_text or candidate.translated_text or "",
                )
                if sim >= DUPLICATE_TEXT_SIMILARITY_THRESHOLD:
                    return candidate

        elif candidate.location and candidate.location == location:
            # Exact location string match (fallback for no GPS)
            return candidate

    return None
