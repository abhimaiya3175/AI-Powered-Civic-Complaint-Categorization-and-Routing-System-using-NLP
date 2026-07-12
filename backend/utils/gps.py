"""
backend/utils/gps.py
====================
GPS / haversine distance utilities.
"""

import math


def convert_dms_to_decimal(dms_value, ref) -> float:
    """Convert GPS DMS tuple (as stored in EXIF) to a decimal degree float."""
    if not isinstance(dms_value, (list, tuple)) or len(dms_value) != 3:
        raise ValueError("Invalid GPS coordinate format")

    def _ratio_to_float(component) -> float:
        # Pillow may expose EXIF rationals as IFDRational, tuples, or plain numbers.
        if hasattr(component, "numerator") and hasattr(component, "denominator"):
            denominator = float(component.denominator) if component.denominator else 1.0
            return float(component.numerator) / denominator
        if isinstance(component, (tuple, list)) and len(component) == 2:
            numerator = float(component[0])
            denominator = float(component[1]) if component[1] else 1.0
            return numerator / denominator
        return float(component)

    degrees = _ratio_to_float(dms_value[0])
    minutes = _ratio_to_float(dms_value[1])
    seconds = _ratio_to_float(dms_value[2])
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two GPS points."""
    earth_radius = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius * c
