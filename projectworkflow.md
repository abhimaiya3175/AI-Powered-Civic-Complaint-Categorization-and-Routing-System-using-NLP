
# Project Workflow and Architecture

## High-level overview
This system has three main parts:

- **Frontend**: React + Vite single-page app in `civic-frontend/`.
- **Backend**: FastAPI app in `main.py` that exposes REST endpoints.
- **Data layer**: PostgreSQL (optional) or SQLite fallback, plus file storage for uploads.

The frontend talks to the backend through `/api` during local dev. Vite proxies `/api` to `http://127.0.0.1:8000`, so the app can call the backend without CORS issues.

## User-facing workflow (citizens)

1. **Open the citizen portal** (frontend).
2. **Capture live location** in the browser. This is required for every submission.
3. **Provide evidence** (optional):
	- Record audio (Kannada/Hindi/English).
	- Capture or upload an image.
	- Add a text note (optional).
4. **Submit complaint**.
5. **Get confirmation** with a complaint ID and initial status.

If image evidence is provided, the backend verifies authenticity using EXIF GPS and timestamp checks. Submissions without valid live location are rejected.

## Admin workflow

1. **Login** with admin credentials (`/login`) to receive a JWT token.
2. **View complaints** with pagination (`/complaints`).
3. **View active complaints on Map** (`/complaints/map`) to visualize non-resolved complaints collectively (no pagination boundaries).
4. **Review evidence** (audio/image) and AI classification.
5. **Verify or edit** complaint (`/complaints/{id}/verify`).
6. **Track stats** (`/complaints/stats`) and complaint status updates.

## Backend processing flow (end-to-end)

This is the exact request path for `POST /submit-complaint` as implemented in main.py.

### 0) Startup initialization (one-time)

- Loads the TF-IDF + Naive Bayes model from `model_bbmp.pkl` or `Models/model_bbmp.pkl`.
- Loads spaCy `en_core_web_sm` (falls back to blank English if missing).
- Loads Whisper `small` model.
- Preloads translation assets in priority order:
  1) IndicTrans2 (with IndicTransToolkit if available)
  2) NLLB fallback
  3) Rotary IndicTrans2 fallback

If translation assets fail to load, translation returns the original text and logs a warning.

### 1) Input validation

- `live_latitude` and `live_longitude` must be valid ranges.
- `live_location_timestamp` must be ISO-8601.
- `language` and `target_language` must be in `kn`, `hi`, or `en`.
- At least one of `file` (audio) or `text_note` must be provided.
- Audio file type and image file type must match allowed extensions and MIME types.
- Max file size is 25 MB for audio and image.

If any of these checks fail, the endpoint returns HTTP 400 with a clear error message.

### 2) Save evidence files

- Audio and image (if present) are saved to `uploads/` with UUID filenames.
- The raw file paths are stored for later DB persistence.

### 3) Transcription (audio only)

- If `language` is `kn`, `hi`, or `en`, the backend tries **Google STT** first.
  - Audio is converted to WAV using pydub (requires FFmpeg on the host).
  - If Google STT fails, it falls back to Whisper.
- For any other language values (not allowed by validation), the request would already be rejected.
- If transcription yields empty text and there is no `text_note`, the request is rejected.

Important: If FFmpeg is missing, transcription fails with a 500 error instructing to install FFmpeg.

### 4) Build the complaint text

- If `text_note` is present, it is appended to the transcribed text.
- The final `transcribed_text` must be non-empty.

### 5) Translation step

- Translation is **always** a separate step after transcription (no Whisper translate task).
- If `target_language` is `en`, the translated text is used for classification.
- If the target language is not English, the backend still generates an English translation for classification.
- If translation assets are unavailable, the original text is used.

### 6) Classification and explanation

- The TF-IDF + Naive Bayes classifier predicts a category.
- An explanation is built from top contributing TF-IDF features.
- If confidence is low or signals are sparse, a zero-shot NLI model can override the category.

If the category is `Non-Civic`, the backend deletes the uploaded files and returns HTTP 400.

### 7) Image authenticity (optional)

- Reads EXIF GPS and timestamp from the image.
- GPS must be within 100 meters of the live location.
- EXIF timestamp must be within 10 minutes.

If any check fails, the request returns HTTP 400.

### 8) Trust tier and status

- Image + valid EXIF + live location: `trust_level=high`, `verification_mode=auto_verified`, `status=Verified`.
- Otherwise: `trust_level=medium`, `verification_mode=manual_review`, `status=pending`.

### 9) Duplicate detection and voting

- Checks for an existing complaint with the same category within 0.5 km in the last 180 days.
- If a duplicate exists:
  - Uploaded files are deleted.
  - If `voter_fingerprint` is provided and not already used, a vote is added.
  - The API returns a `duplicate: true` response instead of creating a new complaint.

### 10) Persistence and timeline

- A new complaint row is inserted into the database.
- A timeline entry is created with status `Reported` and note `Complaint submitted`.
- The response includes category, location, trust level, and explanation details.

## Data storage

- **Database**: PostgreSQL (preferred) or SQLite (`complaints.db`) fallback.
- **File storage**: Uploaded audio/image files in `uploads/`.
- **Logs**: Application logs in `bbmp_complaints.log`.

## Runtime topology (local dev)

- **Frontend**: Vite dev server on `https://localhost:5173`.
- **Backend**: FastAPI on `http://localhost:8000`.
- **Proxy**: Frontend calls `/api/*`, Vite proxies to the backend.
- **DB**: Optional PostgreSQL via Docker Compose, or SQLite fallback.

## Key endpoints summary

- `POST /submit-complaint` (public)
- `POST /login` (public)
- `GET /complaints` (admin)
- `GET /complaints/map` (admin, retrieves non-resolved complaints with GPS)
- `PUT /complaints/{id}/verify` (admin)
- `GET /complaints/stats` (admin)
- `GET /uploads/{filename}` (admin, tokened access)

