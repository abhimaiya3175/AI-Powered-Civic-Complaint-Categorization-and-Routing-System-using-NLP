"""
backend/services/image_service.py
===================================
Background image analysis task: Florence-2 / YOLOv8 + cross-modal verification.
"""

import asyncio
import json
import logging

from image_features import analyze_image
from cross_modal import verify_cross_modal

logger = logging.getLogger("bbmp")


async def run_image_analysis_background(
    complaint_id: int,
    image_path: str,
    nlp_category: str,
) -> None:
    """Background task to run Florence-2 (or YOLOv8) and perform cross-modal verification."""
    from backend.database import SessionLocal
    from backend.models import Complaint

    db = SessionLocal()
    try:
        # 180-second timeout protects against Florence-2 hanging on slow CPUs
        result = await asyncio.wait_for(analyze_image(image_path), timeout=180.0)

        verification = verify_cross_modal(nlp_category, result, existing_trust_level="medium")

        complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
        if not complaint:
            return

        complaint.florence_status = result.get("status", "error")
        if result.get("backend") == "florence":
            complaint.florence_caption = result.get("caption")
            complaint.florence_damaged_object = result.get("damaged_object")
            complaint.florence_problem_type = result.get("problem_type")
            complaint.florence_severity = result.get("severity")
            complaint.florence_evidence = result.get("supporting_evidence")
            complaint.florence_processing_time = result.get("processing_time")
        else:
            # Backward compat for YOLOv8
            complaint.detected_objects = json.dumps(result.get("detections", []))
            complaint.pothole_severity = result.get("severity")

        complaint.image_suggested_category = result.get("suggested_category")
        complaint.florence_all_categories = json.dumps(
            result.get("all_suggested_categories", [])
        )

        # Verification results
        complaint.cross_modal_result = verification.get("verification_result")
        complaint.cross_modal_nlp_category = verification.get("nlp_category")
        complaint.cross_modal_image_category = verification.get("image_category")
        complaint.manual_review_required = verification.get("manual_review_required", False)

        # Update trust level if verification changed it
        if verification.get("verification_result") != "image_unclear":
            complaint.trust_level = verification.get("trust_level", "medium")
            if verification.get("trust_level") == "manual_review":
                complaint.status = "pending"
                complaint.category_mismatch = True  # backward compat

        db.commit()
        logger.info("Background image analysis completed for complaint %d", complaint_id)

    except asyncio.TimeoutError:
        logger.error("Background image analysis timed out for complaint %d", complaint_id)
        complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
        if complaint:
            complaint.florence_status = "timeout"
            db.commit()
    except Exception as exc:
        logger.error("Background image analysis failed for complaint %d: %s", complaint_id, exc)
        complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
        if complaint:
            complaint.florence_status = "error"
            db.commit()
    finally:
        db.close()
