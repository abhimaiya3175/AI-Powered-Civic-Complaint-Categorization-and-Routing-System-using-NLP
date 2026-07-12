"""
backend/utils/exif.py
=====================
EXIF metadata extraction from uploaded images.
"""

from datetime import datetime

from fastapi import HTTPException
from PIL import Image
from PIL.ExifTags import GPSTAGS

from backend.utils.gps import convert_dms_to_decimal


def extract_exif_location_and_time(image_path: str) -> tuple:
    """Return (latitude, longitude, datetime) from image EXIF GPS metadata.

    Raises HTTPException with a descriptive message on any validation failure.
    """
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read image metadata: {exc}")

    if not exif:
        raise HTTPException(
            status_code=400,
            detail="Image must contain EXIF metadata with GPS coordinates and timestamp.",
        )

    gps_info = None
    if hasattr(exif, "get_ifd"):
        try:
            gps_info = exif.get_ifd(34853)
        except Exception:
            gps_info = None

    if not gps_info:
        legacy_gps_info = exif.get(34853)
        if isinstance(legacy_gps_info, dict):
            gps_info = legacy_gps_info

    exif_timestamp = exif.get(36867) or exif.get(306)

    if not gps_info:
        raise HTTPException(status_code=400, detail="Image EXIF GPS metadata is missing.")
    if not exif_timestamp:
        raise HTTPException(status_code=400, detail="Image EXIF timestamp is missing.")

    if isinstance(exif_timestamp, bytes):
        exif_timestamp = exif_timestamp.decode(errors="ignore")

    if not isinstance(gps_info, dict):
        raise HTTPException(status_code=400, detail="Image EXIF GPS metadata is malformed.")

    gps_data = {GPSTAGS.get(tag, tag): value for tag, value in gps_info.items()}
    lat = gps_data.get("GPSLatitude") or gps_data.get(2)
    lat_ref = gps_data.get("GPSLatitudeRef") or gps_data.get(1)
    lon = gps_data.get("GPSLongitude") or gps_data.get(4)
    lon_ref = gps_data.get("GPSLongitudeRef") or gps_data.get(3)

    if isinstance(lat_ref, bytes):
        lat_ref = lat_ref.decode(errors="ignore")
    if isinstance(lon_ref, bytes):
        lon_ref = lon_ref.decode(errors="ignore")

    if not lat or not lon or lat_ref not in ("N", "S") or lon_ref not in ("E", "W"):
        raise HTTPException(status_code=400, detail="Image EXIF GPS metadata is invalid.")

    try:
        image_lat = convert_dms_to_decimal(lat, lat_ref)
        image_lon = convert_dms_to_decimal(lon, lon_ref)
        image_timestamp = datetime.strptime(str(exif_timestamp), "%Y:%m:%d %H:%M:%S")
    except Exception:
        raise HTTPException(status_code=400, detail="Image EXIF metadata is malformed.")

    return image_lat, image_lon, image_timestamp
