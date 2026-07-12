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

# ── Florence-2 caption-context phrases → canonical category ───────────
# Florence-2 captions describe scenes ("wet and muddy road with puddles")
# rather than listing object labels. These contextual phrases are matched
# against <CAPTION> and <MORE_DETAILED_CAPTION> output to catch road/civic
# issues that <OD> misses (e.g. OD sees "building" but caption says "muddy road").
CAPTION_CONTEXT_TO_CATEGORY: dict[str, str] = {
    # Road Repair — caption phrases for damaged / poor road conditions
    "muddy road": "Road Repair",
    "damaged road": "Road Repair",
    "broken road": "Road Repair",
    "rough road": "Road Repair",
    "uneven road": "Road Repair",
    "bumpy road": "Road Repair",
    "road with holes": "Road Repair",
    "dilapidated road": "Road Repair",
    "cracked road": "Road Repair",
    "deteriorated road": "Road Repair",
    "poor road": "Road Repair",
    "bad road": "Road Repair",
    "unpaved road": "Road Repair",
    "dirt road in": "Road Repair",
    "road is in bad": "Road Repair",
    "road is damaged": "Road Repair",
    "road surface": "Road Repair",
    "road is broken": "Road Repair",
    "puddles": "Road Repair",
    "puddle": "Road Repair",
    "mud on road": "Road Repair",
    "mud on the road": "Road Repair",
    "muddy street": "Road Repair",
    "wet and muddy": "Road Repair",
    # Drainage / SWD — flooding / waterlogging
    "flooded road": "Drainage / SWD",
    "flooded street": "Drainage / SWD",
    "water on road": "Drainage / SWD",
    "water on the road": "Drainage / SWD",
    "waterlogged": "Drainage / SWD",
    "water logged": "Drainage / SWD",
    "standing water": "Drainage / SWD",
    "overflowing drain": "Drainage / SWD",
    "water covering": "Drainage / SWD",
    "water pooling": "Drainage / SWD",
    "water filled road": "Drainage / SWD",
    "water fills the road": "Drainage / SWD",
    "large puddles of water": "Drainage / SWD",
    "puddles of water covering": "Drainage / SWD",
    "water covering the ground": "Drainage / SWD",
    # Garbage / Sanitation
    "pile of garbage": "Garbage / Sanitation",
    "garbage pile": "Garbage / Sanitation",
    "litter on": "Garbage / Sanitation",
    "dirty street": "Garbage / Sanitation",
    "waste dump": "Garbage / Sanitation",
    "trash heap": "Garbage / Sanitation",
    "rubbish pile": "Garbage / Sanitation",
    "overflowing bin": "Garbage / Sanitation",
    "waste on road": "Garbage / Sanitation",
    "garbage on road": "Garbage / Sanitation",
    "trash on road": "Garbage / Sanitation",
    "piled up garbage": "Garbage / Sanitation",
    "uncollected waste": "Garbage / Sanitation",
    # Street Light
    "broken lamp": "Street Light",
    "dark street": "Street Light",
    "unlit road": "Street Light",
    "no light": "Street Light",
    "damaged lamp": "Street Light",
    "broken street light": "Street Light",
    # Health / Sanitation
    "stagnant water": "Health / Sanitation",
    "dirty water": "Health / Sanitation",
    "open sewer": "Health / Sanitation",
    "mosquito breeding": "Health / Sanitation",
    # Parks
    "broken playground": "Parks",
    "damaged park": "Parks",
    "overgrown park": "Parks",
    "unmaintained park": "Parks",
    # Traffic
    "traffic jam": "Traffic",
    "traffic congestion": "Traffic",
    "vehicles stuck": "Traffic",
    "heavy traffic": "Traffic",
    "broken signal": "Traffic",
    "traffic signal": "Traffic",
    # Town Planning
    "illegal construction": "Town Planning",
    "unauthorized building": "Town Planning",
    "encroachment": "Town Planning",
    # Water Supply
    "broken pipe": "Water Supply",
    "leaking pipe": "Water Supply",
    "water leak": "Water Supply",
    "no water supply": "Water Supply",
}

# ── Caption-based severity keywords ──────────────────────────────────
# Applied by compute_caption_severity() in image_features.py.
# Checked in order — first matching severity level wins.
SEVERITY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Severe", [
        "collapsed", "dangerous", "hazardous", "flooded", "destroyed",
        "major", "catastrophic", "life-threatening", "emergency", "critical",
        "severely damaged", "completely broken",
    ]),
    ("High", [
        "large", "broken", "blocked", "overflowing", "damaged", "severe",
        "extensive", "significant", "deep", "wide", "big",
        "muddy", "large puddles", "gloomy", "wet and muddy",
        "heavily damaged", "dilapidated", "deteriorated",
    ]),
    ("Medium", [
        "cracked", "leaking", "partial", "moderate", "noticeable",
        "visible", "uneven", "dirty", "clogged",
        "wet", "rough", "puddles", "bumpy", "unpaved",
    ]),
    ("Low", [
        "minor", "small", "slight", "faded", "worn", "thin",
        "narrow", "tiny", "superficial",
    ]),
]

