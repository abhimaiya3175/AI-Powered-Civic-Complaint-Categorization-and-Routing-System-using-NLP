"""
category_registry.py
====================
Single source of truth for civic complaint category strings.

Both the NLP classifier (TF-IDF + NB) and the image analysis module
(Florence-2 / YOLOv8) import from here so equality checks across
text and visual pipelines are always meaningful.
"""

# ── Canonical BBMP complaint categories ──────────────────────────────
CANONICAL_CATEGORIES: list[str] = [
    "Street Light",
    "Garbage / Sanitation",
    "Road Repair",
    "Drainage / SWD",
    "Water Supply",
    "Health / Sanitation",
    "Parks",
    "Parks / Forest",
    "Town Planning",
    "Veterinary",
    "Advertisement",
    "Revenue",
    "Traffic",
    "Others",
    "Non-Civic",
]

# ── Florence-2 visual keywords → canonical category ──────────────────
# Used by image_features.py to map detected objects / region labels
# to the same category strings the NLP classifier produces.
VISUAL_KEYWORD_TO_CATEGORY: dict[str, str] = {
    # Road Repair
    "pothole": "Road Repair",
    "crack": "Road Repair",
    "road damage": "Road Repair",
    "broken road": "Road Repair",
    "road crack": "Road Repair",
    "asphalt damage": "Road Repair",
    # Garbage / Sanitation
    "garbage": "Garbage / Sanitation",
    "trash": "Garbage / Sanitation",
    "waste": "Garbage / Sanitation",
    "litter": "Garbage / Sanitation",
    "garbage pile": "Garbage / Sanitation",
    "dump": "Garbage / Sanitation",
    "rubbish": "Garbage / Sanitation",
    # Street Light
    "streetlight": "Street Light",
    "street light": "Street Light",
    "lamp": "Street Light",
    "light pole": "Street Light",
    "broken light": "Street Light",
    "lamp post": "Street Light",
    # Drainage / SWD
    "drain": "Drainage / SWD",
    "waterlogging": "Drainage / SWD",
    "flooding": "Drainage / SWD",
    "sewage": "Drainage / SWD",
    "storm drain": "Drainage / SWD",
    "water logging": "Drainage / SWD",
    "blocked drain": "Drainage / SWD",
    # Advertisement
    "hoarding": "Advertisement",
    "banner": "Advertisement",
    "billboard": "Advertisement",
    "flex": "Advertisement",
    "poster": "Advertisement",
    "illegal banner": "Advertisement",
    # Parks / Forest
    "park": "Parks",
    "playground": "Parks",
    "tree": "Parks / Forest",
    "overgrown": "Parks / Forest",
    "fallen tree": "Parks / Forest",
    # Water Supply
    "water leak": "Water Supply",
    "pipe": "Water Supply",
    "water pipe": "Water Supply",
    "leaking pipe": "Water Supply",
    "broken pipe": "Water Supply",
    # Veterinary
    "stray dog": "Veterinary",
    "stray animal": "Veterinary",
    "dead animal": "Veterinary",
    "dog": "Veterinary",
    "cow": "Veterinary",
    # Town Planning
    "construction": "Town Planning",
    "building": "Town Planning",
    "encroachment": "Town Planning",
    "illegal construction": "Town Planning",
    # Traffic
    "traffic": "Traffic",
    "signal": "Traffic",
    "traffic light": "Traffic",
    "traffic signal": "Traffic",
    # Health / Sanitation
    "mosquito": "Health / Sanitation",
    "stagnant water": "Health / Sanitation",
    "open defecation": "Health / Sanitation",
}

# ── Caption-based severity keywords ──────────────────────────────────
# Applied by compute_caption_severity() in image_features.py.
# Checked in order — first matching severity level wins.
SEVERITY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Severe", [
        "collapsed", "dangerous", "hazardous", "flooded", "destroyed",
        "major", "catastrophic", "life-threatening", "emergency", "critical",
    ]),
    ("High", [
        "large", "broken", "blocked", "overflowing", "damaged", "severe",
        "extensive", "significant", "deep", "wide", "big",
    ]),
    ("Medium", [
        "cracked", "leaking", "partial", "moderate", "noticeable",
        "visible", "uneven", "dirty", "clogged",
    ]),
    ("Low", [
        "minor", "small", "slight", "faded", "worn", "thin",
        "narrow", "tiny", "superficial",
    ]),
]
