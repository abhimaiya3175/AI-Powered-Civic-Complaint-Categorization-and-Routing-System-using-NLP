"""
image_features.py
=================
Switchable image analysis backend: Florence-2 (default) or YOLOv8n-seg.

Controlled by env var IMAGE_BACKEND=florence|yolo (default: florence).

Florence-2 uses structured grounding tasks (<OD>, <DENSE_REGION_CAPTION>,
<CAPTION>, <MORE_DETAILED_CAPTION>) — NOT prose parsing.

YOLOv8 path is preserved verbatim for rollback safety.

The public interface — analyze_image() — returns the same shape regardless
of backend so main.py never needs to know which engine ran.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from category_registry import (
    CAPTION_CONTEXT_TO_CATEGORY,
    SEVERITY_KEYWORDS,
    VISUAL_KEYWORD_TO_CATEGORY,
)

logger = logging.getLogger("bbmp")

# ====================== CONFIG ======================
IMAGE_BACKEND: str = os.getenv("IMAGE_BACKEND", "florence").strip().lower()

# ====================== SEVERITY RULES (explicit, auditable) ======================

def compute_caption_severity(caption: str, detailed_caption: str = "") -> str:
    """Keyword-based severity derived from Florence-2 caption text.

    Separate from category detection — auditable and tunable.
    Uses SEVERITY_KEYWORDS from category_registry.py.
    """
    combined = f"{caption} {detailed_caption}".lower()
    for severity, keywords in SEVERITY_KEYWORDS:
        if any(kw in combined for kw in keywords):
            return severity
    return "Low"


def _match_visual_category(labels: List[str]) -> Optional[str]:
    """Match a list of detected object labels to a canonical BBMP category.

    Iterates VISUAL_KEYWORD_TO_CATEGORY (longest-match first) and returns
    the first canonical category that matches any label.
    """
    all_cats = _match_all_visual_categories(labels)
    return all_cats[0]["category"] if all_cats else None


def _match_all_visual_categories(labels: List[str]) -> List[Dict[str, Any]]:
    """Match detected labels + captions to ALL relevant BBMP categories.

    Returns a ranked list of {category, matched_keywords, score} dicts.
    Checks both OD/region labels (VISUAL_KEYWORD_TO_CATEGORY) and
    caption-context phrases (CAPTION_CONTEXT_TO_CATEGORY).
    """
    combined = " ".join(label.lower() for label in labels)
    category_hits: Dict[str, List[str]] = {}

    # 1. Match object-detection / region-label keywords
    sorted_od_kw = sorted(VISUAL_KEYWORD_TO_CATEGORY.keys(), key=len, reverse=True)
    for keyword in sorted_od_kw:
        if keyword in combined:
            cat = VISUAL_KEYWORD_TO_CATEGORY[keyword]
            category_hits.setdefault(cat, []).append(keyword)

    # 2. Match caption-context phrases (longer phrases first)
    sorted_caption_kw = sorted(CAPTION_CONTEXT_TO_CATEGORY.keys(), key=len, reverse=True)
    for phrase in sorted_caption_kw:
        if phrase in combined:
            cat = CAPTION_CONTEXT_TO_CATEGORY[phrase]
            category_hits.setdefault(cat, []).append(phrase)

    # Rank by number of matching keywords (more matches = higher confidence)
    return sorted(
        [
            {"category": cat, "matched_keywords": kws, "score": len(kws)}
            for cat, kws in category_hits.items()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  FLORENCE-2 BACKEND
# ══════════════════════════════════════════════════════════════════════

_florence_model: Any = None
_florence_processor: Any = None
_florence_load_attempted: bool = False
_florence_load_lock = None  # Initialized lazily to avoid event-loop issues


def _get_florence_lock():
    """Get or create the asyncio lock for Florence-2 loading."""
    global _florence_load_lock
    if _florence_load_lock is None:
        _florence_load_lock = asyncio.Lock()
    return _florence_load_lock


def load_florence_model() -> bool:
    """Load Florence-2-base model and processor.

    Called lazily on first analyze_image() call — NOT at startup.
    Returns True if model loaded successfully, False otherwise.
    Only marks _florence_load_attempted = True on success so that
    a failed load can be retried (e.g. after upgrading transformers).
    """
    global _florence_model, _florence_processor, _florence_load_attempted

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
    except ImportError:
        logger.warning(
            "transformers package not available for Florence-2. "
            "Image analysis will be skipped."
        )
        return False

    model_name = "microsoft/Florence-2-base"
    try:
        import torch
        from transformers.configuration_utils import PretrainedConfig
        if not hasattr(PretrainedConfig, 'forced_bos_token_id'):
            PretrainedConfig.forced_bos_token_id = None

        logger.info("Loading Florence-2 model: %s …", model_name)
        try:
            _florence_processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True
            )
            _florence_model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float32, local_files_only=True
            )
        except OSError:
            logger.info("Florence-2 not in local cache, downloading from HuggingFace…")
            _florence_processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            _florence_model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float32
            )
        _florence_model.to("cpu")
        _florence_model.eval()
        _florence_load_attempted = True  # Only mark on success
        logger.info("Florence-2 model loaded successfully.")
        return True
    except Exception as exc:
        logger.error("Failed to load Florence-2 model (%s): %s", model_name, exc)
        _florence_model = None
        _florence_processor = None
        return False


def _run_florence_task(image: Image.Image, task_prompt: str) -> Any:
    """Run a single Florence-2 task synchronously."""
    import torch

    inputs = _florence_processor(
        text=task_prompt, images=image, return_tensors="pt"
    )
    # Move to same device as model
    device = next(_florence_model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = _florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            early_stopping=True,
        )

    generated_text = _florence_processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed = _florence_processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed


def _run_florence_inference_sync(image_path: str) -> Dict[str, Any]:
    """Run all Florence-2 tasks synchronously (called via asyncio.to_thread).

    Uses structured grounding tasks for labels, NOT prose parsing.
    """
    t_start = time.perf_counter()
    img = Image.open(image_path).convert("RGB")

    # Task 1: Short caption
    caption = ""
    try:
        result = _run_florence_task(img, "<CAPTION>")
        caption = result.get("<CAPTION>", "")
    except Exception as exc:
        logger.warning("Florence-2 <CAPTION> failed: %s", exc)

    # Task 2: Detailed caption (raw supporting evidence — never parsed for structured fields)
    detailed_caption = ""
    try:
        result = _run_florence_task(img, "<MORE_DETAILED_CAPTION>")
        detailed_caption = result.get("<MORE_DETAILED_CAPTION>", "")
    except Exception as exc:
        logger.warning("Florence-2 <MORE_DETAILED_CAPTION> failed: %s", exc)

    # Task 3: Object Detection (structured labels + bboxes)
    od_labels: List[str] = []
    od_detections: List[Dict[str, Any]] = []
    try:
        result = _run_florence_task(img, "<OD>")
        od_data = result.get("<OD>", {})
        bboxes = od_data.get("bboxes", [])
        labels = od_data.get("labels", [])
        for i, label in enumerate(labels):
            od_labels.append(label)
            bbox = bboxes[i] if i < len(bboxes) else []
            # Normalize bbox to 0-1 range
            if bbox and len(bbox) == 4:
                norm_bbox = [
                    round(bbox[0] / img.width, 6),
                    round(bbox[1] / img.height, 6),
                    round(bbox[2] / img.width, 6),
                    round(bbox[3] / img.height, 6),
                ]
            else:
                norm_bbox = bbox
            od_detections.append({"label": label, "bbox": norm_bbox})
    except Exception as exc:
        logger.warning("Florence-2 <OD> failed: %s", exc)

    # Task 4: Dense Region Captions (region-level labels for problem_type)
    region_labels: List[str] = []
    try:
        result = _run_florence_task(img, "<DENSE_REGION_CAPTION>")
        drc_data = result.get("<DENSE_REGION_CAPTION>", {})
        region_labels = drc_data.get("labels", [])
    except Exception as exc:
        logger.warning("Florence-2 <DENSE_REGION_CAPTION> failed: %s", exc)

    # Derive structured fields from grounding + captions
    all_labels = od_labels + region_labels + [caption, detailed_caption]

    # Multi-category matching: check OD labels AND caption context phrases
    all_categories = _match_all_visual_categories(all_labels)
    suggested_category = all_categories[0]["category"] if all_categories else None

    # damaged_object: prefer a civic-meaningful label over raw COCO labels.
    # Check if any OD label maps to a known civic category; if so, use a
    # human-readable civic description. Otherwise fall back to the raw OD label.
    _OD_CIVIC_MAP = {
        "pothole": "Pothole / Road Damage",
        "street light": "Street Light",
        "garbage": "Garbage Pile",
        "waste": "Waste Pile",
        "sewage": "Sewage / Drain Overflow",
        "drain": "Blocked Drain",
        "flood": "Flooding / Waterlogging",
        "hoarding": "Illegal Hoarding",
        "dog": "Stray Dog",
        "mosquito": "Mosquito Breeding",
    }
    damaged_object = None
    for raw_label in od_labels:
        civic = _OD_CIVIC_MAP.get(raw_label.lower().strip())
        if civic:
            damaged_object = civic
            break
    if damaged_object is None and od_labels:
        # No civic map hit — use top matched keyword as civic description
        if all_categories:
            top_kws = all_categories[0]["matched_keywords"]
            damaged_object = top_kws[0].replace(" of water covering", "").replace(" the ground", "").title() if top_kws else od_labels[0]
        else:
            damaged_object = od_labels[0]  # raw COCO label as last resort

    # problem_type: canonical BBMP category name (human-readable), not a raw phrase
    # matched_evidence: the raw keyword phrase that triggered detection
    problem_type = None
    matched_evidence = None
    if all_categories:
        problem_type = all_categories[0]["category"]  # e.g. "Drainage / SWD"
        matched_evidence = all_categories[0]["matched_keywords"][0] if all_categories[0]["matched_keywords"] else None

    # Civic fallback: if no phrases matched but caption mentions road/street/vehicle,
    # the image is likely civic-relevant — return "Others" instead of None.
    if suggested_category is None:
        _fallback_cues = ["road", "street", "vehicle", "pothole", "drain", "water", "garbage",
                          "light", "building", "construction", "park", "dog", "mosquito"]
        combined_lower = (caption + " " + detailed_caption).lower()
        if any(cue in combined_lower for cue in _fallback_cues):
            suggested_category = "Others"
            problem_type = "Others"
            matched_evidence = "generic civic scene detected"
            if damaged_object is None and od_labels:
                damaged_object = od_labels[0]

    # Severity from caption text (explicit rule layer)
    severity = compute_caption_severity(caption, detailed_caption)

    processing_time = round(time.perf_counter() - t_start, 4)

    return {
        "backend": "florence",
        "status": "success",
        "caption": caption,
        "supporting_evidence": detailed_caption,
        "damaged_object": damaged_object,
        "problem_type": problem_type,
        "matched_evidence": matched_evidence,
        "suggested_category": suggested_category,
        "all_suggested_categories": all_categories,
        "severity": severity,
        "detections": od_detections,
        "processing_time": processing_time,
    }


async def _analyze_image_florence(image_path: str) -> Dict[str, Any]:
    """Florence-2 async entry point with lazy loading."""
    global _florence_model, _florence_processor

    # Lazy load on first call
    if not _florence_load_attempted:
        async with _get_florence_lock():
            if not _florence_load_attempted:
                loaded = await asyncio.to_thread(load_florence_model)
                if not loaded:
                    return {
                        "backend": "florence",
                        "status": "unavailable",
                        "reason": "Florence-2 model failed to load.",
                    }

    if _florence_model is None or _florence_processor is None:
        return {
            "backend": "florence",
            "status": "unavailable",
            "reason": "Florence-2 model not loaded.",
        }

    try:
        result = await asyncio.to_thread(_run_florence_inference_sync, image_path)
        return result
    except Exception as exc:
        logger.error("Florence-2 inference failed: %s", exc)
        return {
            "backend": "florence",
            "status": "error",
            "reason": str(exc),
        }


# ══════════════════════════════════════════════════════════════════════
#  YOLO BACKEND (preserved verbatim for rollback)
# ══════════════════════════════════════════════════════════════════════

YOLO_MODEL_PATH_CANDIDATES: List[str] = [
    os.getenv("YOLO_MODEL_PATH", "").strip(),
    "Models/civic_multiclass_seg_best.pt",
    "models/civic_multiclass_seg_best.pt",
]

YOLO_INPUT_SIZE: int = 640

DETECTION_CLASS_TO_CATEGORY: Dict[str, str] = {
    "pothole": "Road Repair",
    "garbage_pile": "Garbage / Sanitation",
    "broken_streetlight": "Street Light",
    "waterlogging": "Drainage / SWD",
    "damaged_drain": "Drainage / SWD",
    "illegal_hoarding": "Advertisement",
    "overgrown_park": "Parks / Forest",
    "water_leak": "Water Supply",
}

IMAGE_SEVERITY_THRESHOLDS: Dict[str, float] = {
    "Severe": 0.15,
    "High": 0.07,
    "Medium": 0.02,
    "Low": 0.0,
}

yolo_model: Any = None


def load_yolo_model() -> None:
    """Load YOLOv8n-seg checkpoint once at startup."""
    global yolo_model

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning(
            "ultralytics package not installed — image analysis stage will be skipped. "
            "Install with: pip install ultralytics"
        )
        return

    for candidate in YOLO_MODEL_PATH_CANDIDATES:
        if not candidate:
            continue
        resolved = Path(candidate).resolve()
        if resolved.exists():
            try:
                yolo_model = YOLO(str(resolved))
                logger.info("YOLOv8n-seg model loaded from %s", resolved)
                return
            except Exception as exc:
                logger.error("Failed to load YOLO model from %s: %s", resolved, exc)

    logger.warning(
        "YOLOv8n-seg checkpoint not found at any candidate path (%s). "
        "Image analysis stage will be skipped. Place your fine-tuned weights "
        "at Models/civic_multiclass_seg_best.pt.",
        ", ".join(p for p in YOLO_MODEL_PATH_CANDIDATES if p),
    )


def _compute_yolo_severity(mask_area_ratio: float) -> str:
    """Bucket a mask-area ratio into a severity level."""
    for severity, threshold in IMAGE_SEVERITY_THRESHOLDS.items():
        if mask_area_ratio >= threshold:
            return severity
    return "Low"


def _polygon_area(polygon: List[List[float]]) -> float:
    """Compute area of a polygon using the shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def _run_yolo_inference_sync(image_path: str) -> Dict[str, Any]:
    """Run YOLOv8n-seg inference synchronously."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    scale = min(YOLO_INPUT_SIZE / orig_w, YOLO_INPUT_SIZE / orig_h, 1.0)
    inf_w = int(orig_w * scale)
    inf_h = int(orig_h * scale)
    if scale < 1.0:
        img_resized = img.resize((inf_w, inf_h), Image.LANCZOS)
    else:
        img_resized = img
        inf_w, inf_h = orig_w, orig_h

    results = yolo_model.predict(
        source=img_resized,
        imgsz=YOLO_INPUT_SIZE,
        device="cpu",
        verbose=False,
        conf=0.25,
    )

    detections: List[Dict[str, Any]] = []
    image_area_px = inf_w * inf_h

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        masks = result.masks

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            confidence = round(float(boxes.conf[i].item()), 4)
            cls_name = result.names.get(cls_id, f"class_{cls_id}")

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox_normalized = [
                round(x1 / inf_w, 6),
                round(y1 / inf_h, 6),
                round(x2 / inf_w, 6),
                round(y2 / inf_h, 6),
            ]

            mask_polygon_normalized: List[List[float]] = []
            mask_area_ratio = 0.0
            if masks is not None and i < len(masks.xy):
                raw_polygon = masks.xy[i].tolist()
                if raw_polygon:
                    mask_polygon_normalized = [
                        [round(pt[0] / inf_w, 6), round(pt[1] / inf_h, 6)]
                        for pt in raw_polygon
                    ]
                    mask_area_px = _polygon_area(raw_polygon)
                    mask_area_ratio = mask_area_px / image_area_px if image_area_px > 0 else 0.0

            severity = _compute_yolo_severity(mask_area_ratio)

            detections.append({
                "class": cls_name,
                "confidence": confidence,
                "bbox": bbox_normalized,
                "mask_polygon": mask_polygon_normalized,
                "mask_area_ratio": round(mask_area_ratio, 6),
                "severity": severity,
            })

    return {
        "detections": detections,
        "image_width": orig_w,
        "image_height": orig_h,
    }


async def _analyze_image_yolo(image_path: str) -> Dict[str, Any]:
    """YOLOv8 async entry point."""
    if yolo_model is None:
        logger.warning(
            "YOLOv8 model not loaded — skipping image analysis for %s",
            image_path,
        )
        return {
            "backend": "yolo",
            "status": "unavailable",
            "reason": "YOLOv8 model not loaded.",
            "detections": [],
            "severity": None,
        }

    result = await asyncio.to_thread(_run_yolo_inference_sync, image_path)
    detections = result.get("detections", [])

    if not detections:
        overall_severity = "Clear"
    else:
        severity_order = {"Low": 0, "Medium": 1, "High": 2, "Severe": 3}
        overall_severity = max(
            (d["severity"] for d in detections),
            key=lambda s: severity_order.get(s, -1),
        )

    # Map top detection to a category
    suggested_category = None
    if detections:
        top_det = max(detections, key=lambda d: d["confidence"])
        suggested_category = DETECTION_CLASS_TO_CATEGORY.get(top_det["class"])

    return {
        "backend": "yolo",
        "status": "success",
        "detections": detections,
        "severity": overall_severity,
        "suggested_category": suggested_category,
        "caption": None,
        "supporting_evidence": None,
        "damaged_object": detections[0]["class"] if detections else None,
        "problem_type": detections[0]["class"] if detections else None,
        "processing_time": 0.0,
    }


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC API — stable interface for main.py
# ══════════════════════════════════════════════════════════════════════

async def analyze_image(image_path: str) -> Dict[str, Any]:
    """Analyze an image using the configured backend (Florence-2 or YOLOv8).

    Returns a dict with at minimum:
        backend: "florence" | "yolo"
        status: "success" | "unavailable" | "error"
        suggested_category: str | None
        severity: str | None
        caption: str | None
        supporting_evidence: str | None
        damaged_object: str | None
        problem_type: str | None
        detections: list
        processing_time: float
    """
    if IMAGE_BACKEND == "yolo":
        return await _analyze_image_yolo(image_path)
    else:
        return await _analyze_image_florence(image_path)
