"""
image_features.py
=================
YOLOv8n-seg pothole / road-damage detection module.

Architectural pattern: module-level singleton model, loaded once at startup,
inference offloaded via asyncio.to_thread() — identical to how whisper_model,
nlp (spaCy), and clf (sklearn) are handled in main.py.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger("bbmp")

# ====================== CONSTANTS ======================

# TODO: Replace with the path to your fine-tuned YOLOv8n-seg checkpoint.
YOLO_MODEL_PATH_CANDIDATES: List[str] = [
    os.getenv("YOLO_MODEL_PATH", "").strip(),
    "Models/civic_multiclass_seg_best.pt",
    "models/civic_multiclass_seg_best.pt",
]

YOLO_INPUT_SIZE: int = 640  # Max dimension for inference (controls latency on CPU)

# Mapping from YOLO model detection classes to BBMP Complaint Categories
DETECTION_CLASS_TO_CATEGORY: Dict[str, str] = {
    "pothole": "Road Repair",
    "garbage_pile": "Garbage / Sanitation",
    "broken_streetlight": "Street Light",
    "waterlogging": "Drainage / SWD",
    "damaged_drain": "Drainage / SWD",
    "illegal_hoarding": "Advertisement",
    "overgrown_park": "Parks / Forest",
    "water_leak": "Water Supply"
}

# Severity thresholds: mask_area / image_area ratio -> severity bucket.
# These are cumulative — the first matching threshold wins (checked high-to-low).
IMAGE_SEVERITY_THRESHOLDS: Dict[str, float] = {
    "Severe": 0.15,   # >= 15% of image covered by damage
    "High":   0.07,   # >= 7%
    "Medium": 0.02,   # >= 2%
    "Low":    0.0,    # any detection at all
}

# ====================== MODULE-LEVEL MODEL ======================

yolo_model: Any = None


def load_yolo_model() -> None:
    """Load YOLOv8n-seg checkpoint once at startup.

    Sets the module-level ``yolo_model`` global. If the checkpoint file is
    missing or Ultralytics is not installed, logs a warning and leaves
    ``yolo_model`` as None — the pipeline will skip image analysis gracefully.
    """
    global yolo_model

    try:
        from ultralytics import YOLO  # type: ignore
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


# ====================== INFERENCE ======================

def _compute_severity(mask_area_ratio: float) -> str:
    """Bucket a mask-area ratio into a severity level."""
    for severity, threshold in IMAGE_SEVERITY_THRESHOLDS.items():
        if mask_area_ratio >= threshold:
            return severity
    return "Low"


def _polygon_area(polygon: List[List[float]]) -> float:
    """Compute area of a polygon using the shoelace formula.

    ``polygon`` is a list of [x, y] pairs (in any coordinate space).
    """
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
    """Run YOLOv8n-seg inference synchronously (called via asyncio.to_thread).

    Returns detection results with all bbox/mask coordinates **normalized to
    0–1 range** so the frontend can scale them to any rendered image size.
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Resize for inference (preserving aspect ratio, max dimension = YOLO_INPUT_SIZE)
    scale = min(YOLO_INPUT_SIZE / orig_w, YOLO_INPUT_SIZE / orig_h, 1.0)
    inf_w = int(orig_w * scale)
    inf_h = int(orig_h * scale)
    if scale < 1.0:
        img_resized = img.resize((inf_w, inf_h), Image.LANCZOS)
    else:
        img_resized = img
        inf_w, inf_h = orig_w, orig_h

    # Run inference
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
            # Class and confidence
            cls_id = int(boxes.cls[i].item())
            confidence = round(float(boxes.conf[i].item()), 4)
            cls_name = result.names.get(cls_id, f"class_{cls_id}")

            # Bounding box — normalize to 0–1 range
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox_normalized = [
                round(x1 / inf_w, 6),
                round(y1 / inf_h, 6),
                round(x2 / inf_w, 6),
                round(y2 / inf_h, 6),
            ]

            # Segmentation mask polygon — normalize to 0–1 range
            mask_polygon_normalized: List[List[float]] = []
            mask_area_ratio = 0.0
            if masks is not None and i < len(masks.xy):
                raw_polygon = masks.xy[i].tolist()  # [[x,y], [x,y], ...]
                if raw_polygon:
                    mask_polygon_normalized = [
                        [round(pt[0] / inf_w, 6), round(pt[1] / inf_h, 6)]
                        for pt in raw_polygon
                    ]
                    # Area ratio uses inference-space pixels (scale-invariant ratio)
                    mask_area_px = _polygon_area(raw_polygon)
                    mask_area_ratio = mask_area_px / image_area_px if image_area_px > 0 else 0.0

            severity = _compute_severity(mask_area_ratio)

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


async def analyze_image(image_path: str) -> Dict[str, Any]:
    """Async entry point for image-based pothole/road-damage detection.

    Follows the same asyncio.to_thread() pattern used for whisper_model.transcribe().

    Returns:
        dict with keys: detections, severity, annotated_image_path
        - severity is None if model not loaded (analysis didn't run)
        - severity is "Clear" if model ran but found no detections
        - severity is Low/Medium/High/Severe based on worst detection
    """
    # Early return if model not loaded (checkpoint missing / import failed)
    if yolo_model is None:
        logger.warning(
            "YOLOv8 model not loaded — skipping image analysis for %s",
            image_path,
        )
        return {
            "detections": [],
            "severity": None,
            "annotated_image_path": None,
        }

    result = await asyncio.to_thread(_run_yolo_inference_sync, image_path)
    detections = result.get("detections", [])

    # Determine overall severity
    if not detections:
        overall_severity = "Clear"
    else:
        # Pick the worst (highest) severity across all detections
        severity_order = {"Low": 0, "Medium": 1, "High": 2, "Severe": 3}
        overall_severity = max(
            (d["severity"] for d in detections),
            key=lambda s: severity_order.get(s, -1),
        )

    return {
        "detections": detections,
        "severity": overall_severity,
        "annotated_image_path": None,
    }
