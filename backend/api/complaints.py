"""
backend/api/complaints.py
==========================
Complaint submission, listing, voting, timeline, and media serving routes.
"""

import asyncio
import math
import mimetypes
import os
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile,
)
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.config import SECRET_KEY, ALGORITHM, logger
from backend.core.dependencies import get_db
from backend.core.security import get_current_user
from backend.models import Complaint, ComplaintTimeline, ComplaintVote
from backend.schemas import VerifyRequest, VoteRequest
from backend.services.complaint_service import submit_complaint as _submit_complaint
from backend.services.image_service import run_image_analysis_background

router = APIRouter()


# ---------- Submit Complaint ----------
@router.post("/submit-complaint")
async def submit_complaint(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    live_latitude: float = Form(...),
    live_longitude: float = Form(...),
    live_location_timestamp: str = Form(...),
    text_note: str = Form(default=""),
    language: str = Form(default="en"),
    target_language: str = Form(default="en"),
    voter_fingerprint: str = Form(default=""),
    db: Session = Depends(get_db),
):
    """
    Full NLP pipeline: receive audio → transcribe (Whisper) →
    translate (dedicated NLP model) → classify → save to DB.
    """
    return await _submit_complaint(
        file=file,
        image=image,
        live_latitude=live_latitude,
        live_longitude=live_longitude,
        live_location_timestamp=live_location_timestamp,
        text_note=text_note,
        language=language,
        target_language=target_language,
        voter_fingerprint=voter_fingerprint,
        db=db,
        app_state=request.app.state,
    )


# ---------- Audio / Image File Serving ----------
@router.get("/uploads/{filename}")
async def serve_audio(
    filename: str,
    request: Request,
    token: Optional[str] = Query(default=None),
):
    """Serve uploaded audio/image files to authenticated users."""
    access_token = token
    if not access_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            access_token = auth_header.split(" ", 1)[1].strip()

    if not access_token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    file_path = os.path.join("uploads", filename)
    if not os.path.exists(file_path):
        logger.warning("Audio file not found: %s", filename)
        raise HTTPException(status_code=404, detail="Audio file not found.")

    media_type, _ = mimetypes.guess_type(file_path)
    if not media_type:
        media_type = "application/octet-stream"

    logger.info("Serving audio file: %s", filename)
    return FileResponse(path=file_path, media_type=media_type, filename=filename)


# ---------- List Complaints (Paginated, JWT-protected) ----------
@router.get("/complaints")
async def get_complaints(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=100),
    category_mismatch: Optional[bool] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    query = db.query(Complaint)
    if category_mismatch is True:
        query = query.filter(Complaint.category_mismatch == True)

    total = query.count()
    pages = math.ceil(total / size) if total > 0 else 1
    offset = (page - 1) * size
    items = query.order_by(
        Complaint.votes.desc(),
        Complaint.created_at.desc(),
    ).offset(offset).limit(size).all()

    logger.info(
        "GET /complaints page=%d size=%d total=%d mismatch=%s",
        page, size, total, category_mismatch,
    )

    formatted_items = []
    for c in items:
        c_dict = {column.name: getattr(c, column.name) for column in c.__table__.columns}
        c_dict["florence_analysis"] = {
            "status": c.florence_status,
            "caption": c.florence_caption,
            "damaged_object": c.florence_damaged_object,
            "problem_type": c.florence_problem_type,
            "severity": c.florence_severity,
            "supporting_evidence": c.florence_evidence,
            "processing_time": c.florence_processing_time,
        }
        c_dict["cross_modal"] = {
            "nlp_category": c.cross_modal_nlp_category or c.category,
            "image_category": c.cross_modal_image_category,
            "verification_result": c.cross_modal_result,
            "trust_level": c.trust_level,
            "manual_review_required": c.manual_review_required,
        }
        formatted_items.append(c_dict)

    return {"items": formatted_items, "total": total, "page": page, "size": size, "pages": pages}


# ---------- Verify / Edit Complaint (HITL, JWT-protected) ----------
@router.put("/complaints/{id}/verify")
def verify_complaint(
    id: int,
    request: VerifyRequest,
    db: Session = Depends(get_db),
):
    """HITL: admin verifies or edits a complaint's category/status."""
    complaint = db.query(Complaint).filter(Complaint.id == id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    if request.category:
        complaint.category = request.category
    if request.status:
        complaint.status = request.status

    timeline_entry = ComplaintTimeline(
        complaint_id=id,
        status=request.status,
        note=(request.note or "").strip() or None,
    )
    db.add(timeline_entry)
    db.commit()
    db.refresh(complaint)
    logger.info(
        "Complaint #%d updated — status=%s, category=%s",
        id, complaint.status, complaint.category,
    )

    return {
        "id": complaint.id,
        "category": complaint.category,
        "location": complaint.location,
        "translated_text": complaint.translated_text,
        "status": complaint.status,
    }


# ---------- Vote on Complaint ----------
@router.post("/complaints/{id}/vote")
def vote_complaint(id: int, body: VoteRequest, db: Session = Depends(get_db)):
    """Upvote a complaint. Idempotent — one vote per fingerprint per complaint."""
    complaint = db.query(Complaint).filter(Complaint.id == id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    fingerprint = (body.voter_fingerprint or "").strip()
    if not fingerprint:
        raise HTTPException(status_code=400, detail="voter_fingerprint is required")

    existing_vote = db.query(ComplaintVote).filter(
        ComplaintVote.complaint_id == id,
        ComplaintVote.voter_fingerprint == fingerprint,
    ).first()

    if existing_vote:
        return {"already_voted": True, "votes": complaint.votes or 0}

    db.add(ComplaintVote(complaint_id=id, voter_fingerprint=fingerprint))
    complaint.votes = (complaint.votes or 0) + 1
    db.commit()
    db.refresh(complaint)
    return {"already_voted": False, "votes": complaint.votes}


# ---------- Public Complaints Listing (no JWT) ----------
@router.get("/complaints/public")
def get_public_complaints(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=12, ge=1, le=100),
    category: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    sort: str = Query(default="latest"),
    voter_fingerprint: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """Public-facing complaint listing — excludes Resolved, no auth needed."""
    query = db.query(Complaint).filter(Complaint.status != "Resolved")
    if category and category != "all":
        query = query.filter(Complaint.category == category)
    if status and status != "all":
        query = query.filter(Complaint.status == status)
    if sort == "most_voted":
        query = query.order_by(Complaint.votes.desc(), Complaint.created_at.desc())
    else:
        query = query.order_by(Complaint.created_at.desc())

    total = query.count()
    pages = math.ceil(total / size) if total else 1
    items = query.offset((page - 1) * size).limit(size).all()

    total_votes = db.query(func.sum(Complaint.votes)).filter(Complaint.status != "Resolved").scalar() or 0

    voted_ids = set()
    if voter_fingerprint:
        item_ids = [c.id for c in items]
        if item_ids:
            votes = db.query(ComplaintVote.complaint_id).filter(
                ComplaintVote.voter_fingerprint == voter_fingerprint,
                ComplaintVote.complaint_id.in_(item_ids),
            ).all()
            voted_ids = {v[0] for v in votes}

    return {
        "items": [
            {
                "id": c.id,
                "category": c.category,
                "location": c.location,
                "status": c.status,
                "votes": c.votes or 0,
                "translated_text": c.translated_text,
                "original_text": c.original_text,
                "language": c.language,
                "trust_level": c.trust_level,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "live_latitude": c.live_latitude,
                "live_longitude": c.live_longitude,
                "voted": c.id in voted_ids,
            }
            for c in items
        ],
        "total": total,
        "page": page,
        "pages": pages,
        "size": size,
        "total_votes": total_votes,
    }


# ---------- Resolved Complaints Archive (no JWT) ----------
@router.get("/complaints/resolved")
def get_resolved_complaints(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=12, ge=1, le=100),
    category: Optional[str] = Query(default=None),
    voter_fingerprint: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """Return only resolved complaints for archive tab."""
    query = db.query(Complaint).filter(Complaint.status == "Resolved")
    if category and category != "all":
        query = query.filter(Complaint.category == category)
    query = query.order_by(Complaint.created_at.desc())

    total = query.count()
    pages = math.ceil(total / size) if total else 1
    items = query.offset((page - 1) * size).limit(size).all()

    total_votes = db.query(func.sum(Complaint.votes)).filter(Complaint.status == "Resolved").scalar() or 0

    voted_ids = set()
    if voter_fingerprint:
        item_ids = [c.id for c in items]
        if item_ids:
            votes = db.query(ComplaintVote.complaint_id).filter(
                ComplaintVote.voter_fingerprint == voter_fingerprint,
                ComplaintVote.complaint_id.in_(item_ids),
            ).all()
            voted_ids = {v[0] for v in votes}

    return {
        "items": [
            {
                "id": c.id,
                "category": c.category,
                "location": c.location,
                "status": c.status,
                "votes": c.votes or 0,
                "translated_text": c.translated_text,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "voted": c.id in voted_ids,
            }
            for c in items
        ],
        "total": total,
        "page": page,
        "pages": pages,
        "total_votes": total_votes,
    }


# ---------- Complaint Timeline (no JWT) ----------
@router.get("/complaints/{id}/timeline")
def get_complaint_timeline(id: int, db: Session = Depends(get_db)):
    """Return the status timeline for a complaint."""
    complaint = db.query(Complaint).filter(Complaint.id == id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    entries = (
        db.query(ComplaintTimeline)
        .filter(ComplaintTimeline.complaint_id == id)
        .order_by(ComplaintTimeline.created_at.asc())
        .all()
    )
    return {
        "complaint_id": id,
        "current_status": complaint.status,
        "timeline": [
            {
                "status": e.status,
                "note": e.note,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in entries
        ],
    }


# ---------- Admin: Re-analyze Image ----------
@router.post("/complaints/{id}/reanalyze")
async def reanalyze_complaint_image(
    id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Admin action: re-run Florence-2 on an existing complaint's image."""
    complaint = db.query(Complaint).filter(Complaint.id == id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    if not complaint.image_path:
        raise HTTPException(status_code=400, detail="Complaint has no image")

    complaint.florence_status = "processing"
    db.commit()

    asyncio.create_task(
        run_image_analysis_background(complaint.id, complaint.image_path, complaint.category)
    )
    logger.info("Admin %s triggered image re-analysis for complaint %d", current_user, id)
    return {"msg": "Image re-analysis started in the background", "status": "processing"}
