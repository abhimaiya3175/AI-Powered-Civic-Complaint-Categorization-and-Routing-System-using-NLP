# pipeline/extractor.py
"""
Named-entity location extraction using spaCy.
Returns raw place strings found in the text.
Geocoding stub included – plug in geopy/Nominatim if needed.
"""

from dataclasses import dataclass, field
import spacy


LOCATION_LABELS = {"GPE", "LOC", "FAC"}


@dataclass
class LocationResult:
    raw_places: list[str]           # e.g. ["Bengaluru", "MG Road"]
    display:    str                 # comma-joined or "Not detected"
    lat:        float | None = None
    lon:        float | None = None


class LocationExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

    def extract(self, text: str) -> LocationResult:
        """
        Pull GPE / LOC / FAC entities from English text.
        Geocoding is best-effort: falls back gracefully if unavailable.
        """
        doc    = self.nlp(text)
        places = [ent.text for ent in doc.ents if ent.label_ in LOCATION_LABELS]
        unique = list(dict.fromkeys(places))  # deduplicate, preserve order

        result = LocationResult(
            raw_places=unique,
            display=", ".join(unique) if unique else "Not detected",
        )

        # --- Optional geocoding (uncomment + pip install geopy) ---
        # if unique:
        #     from geopy.geocoders import Nominatim
        #     from geopy.exc import GeocoderTimedOut
        #     geolocator = Nominatim(user_agent="bbmp_pipeline")
        #     try:
        #         loc = geolocator.geocode(unique[0], timeout=5)
        #         if loc:
        #             result.lat = loc.latitude
        #             result.lon = loc.longitude
        #     except GeocoderTimedOut:
        #         pass

        return result