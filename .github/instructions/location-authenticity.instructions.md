---
description: "Use when implementing complaint submission, media upload, geolocation, map display, validation, or anti-fraud checks. Enforces real-time user location capture and GPS-authenticated image evidence; forbids dummy or placeholder location values."
name: "Location Authenticity Rules"
applyTo:
  - "main.py"
  - "civic-frontend/src/**/*.js"
  - "civic-frontend/src/**/*.jsx"
---
# Location Authenticity Rules

- Always capture live device location during complaint submission (latitude, longitude, and capture timestamp).
- Always request location permission explicitly and fail safely when denied; show a clear user message and do not insert fallback values such as "Unknown" or hardcoded coordinates.
- The complaint UI must show two explicit image evidence actions side by side (no dropdown):
  - Capture Photo (device camera)
  - Upload from Gallery
- Image uploads used as evidence must include valid EXIF GPS metadata (latitude, longitude, timestamp).
- Reject image evidence when GPS metadata is missing, malformed, stale, or inconsistent with captured live location.
- Validate metadata on the backend as the source of truth; do not rely only on frontend checks.
- Compare live location and image GPS metadata using a strict 100-meter tolerance radius and reject mismatches.
- Enforce image recency by EXIF timestamp: reject any image older than 10 minutes from submission time.
- Persist original captured coordinates and validated metadata values for auditing.
- Never generate or inject dummy location text, fake GPS tags, or placeholder coordinates in responses or storage.
- When authenticity checks fail, return explicit error details and remediation guidance (retake image with location enabled, grant permission, retry).
- Trust classification rules are mandatory:
  - Image + Live Location => High trust (auto-verified)
  - Text/Audio + Live Location => Medium trust (manual review)
  - No Live Location => Submission rejected
- Add/maintain tests covering:
  - Permission denied flow
  - Camera capture and gallery upload UI visibility (no dropdown)
  - Missing EXIF GPS
  - Invalid EXIF GPS
  - EXIF timestamp older than 10 minutes
  - Metadata/live-location mismatch
  - Successful valid-location submission within 100 meters
  - Trust tier assignment (High/Medium/Rejected)
