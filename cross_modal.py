"""
cross_modal.py
==============
Cross-Modal Verification Engine.

Compares the NLP-predicted department with the Florence-2 (or YOLOv8)
visual analysis result. Both sides use canonical category strings from
category_registry.py, so equality checks are always meaningful.
"""

import logging
from typing import Any, Dict, Optional

from category_registry import CANONICAL_CATEGORIES

logger = logging.getLogger("bbmp")


def verify_cross_modal(
    nlp_category: str,
    image_result: Dict[str, Any],
    existing_trust_level: str = "medium",
) -> Dict[str, Any]:
    """Compare NLP predicted department with image analysis result.

    Parameters
    ----------
    nlp_category : str
        The category predicted by the NLP pipeline (TF-IDF or zero-shot).
    image_result : dict
        The result dict from analyze_image(). Must contain at minimum:
        ``status``, ``suggested_category``.
    existing_trust_level : str
        The trust level assigned by the NLP/EXIF pipeline before image
        analysis. Used as the fallback when image analysis is inconclusive.

    Returns
    -------
    dict with keys:
        categories_match : bool
        nlp_category : str
        image_category : str | None
        verification_result : "match" | "mismatch" | "image_unclear"
        trust_level : "high" | "medium" | "manual_review"
        manual_review_required : bool
    """
    # Default: image analysis didn't produce a usable category
    result = {
        "categories_match": False,
        "nlp_category": nlp_category,
        "image_category": None,
        "verification_result": "image_unclear",
        "trust_level": existing_trust_level,
        "manual_review_required": False,
    }

    # Guard: image analysis didn't run or failed
    if image_result is None:
        return result

    status = image_result.get("status", "")
    if status != "success":
        logger.info(
            "Cross-modal: image status=%s, treating as image_unclear", status
        )
        return result

    image_category = image_result.get("suggested_category")
    result["image_category"] = image_category

    # Guard: image analysis ran but couldn't identify a category
    if image_category is None:
        logger.info(
            "Cross-modal: image analysis succeeded but no category suggested, "
            "treating as image_unclear"
        )
        return result

    # Collect all image-suggested categories for multi-match
    all_image_categories = [
        c["category"]
        for c in image_result.get("all_suggested_categories", [])
        if isinstance(c, dict) and c.get("category")
    ]
    if image_category and image_category not in all_image_categories:
        all_image_categories.insert(0, image_category)
    result["all_image_categories"] = all_image_categories

    # Validate both categories are canonical
    if nlp_category not in CANONICAL_CATEGORIES:
        logger.warning(
            "Cross-modal: NLP category '%s' not in canonical list", nlp_category
        )
    for img_cat in all_image_categories:
        if img_cat not in CANONICAL_CATEGORIES:
            logger.warning(
                "Cross-modal: image category '%s' not in canonical list",
                img_cat,
            )

    # Compare — NLP category matches if it appears in ANY of the image categories
    if nlp_category == image_category or nlp_category in all_image_categories:
        result["categories_match"] = True
        result["verification_result"] = "match"
        result["trust_level"] = "high"
        result["manual_review_required"] = False
        logger.info(
            "Cross-modal MATCH: NLP=%s, Image=%s (all=%s) → trust=high",
            nlp_category,
            image_category,
            all_image_categories,
        )
    else:
        result["categories_match"] = False
        result["verification_result"] = "mismatch"
        result["trust_level"] = "manual_review"
        result["manual_review_required"] = True
        logger.info(
            "Cross-modal MISMATCH: NLP=%s, Image=%s (all=%s) → trust=manual_review",
            nlp_category,
            image_category,
            all_image_categories,
        )

    return result
