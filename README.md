<div align="center">

# 🏛️ BBMP Civic Complaint System

**A Multilingual, AI-Powered Civic Complaint Platform with Live Location and Evidence Authenticity Checks**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)](https://vite.dev)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 **Voice Capture** | Citizens record complaints in **Kannada**, **Hindi**, or **English** |
| 📍 **Location Tagging** | Citizens can auto-detect GPS OR pick/drag a marker on an interactive Leaflet map (with address search and geocoding) |
| 📷 **Image Authenticity** | Camera/gallery image evidence validated via EXIF GPS + timestamp checks |
| 📸 **Florence-2 Image Analysis** | Structured vision-language analysis (using `<OD>`, `<DENSE_REGION_CAPTION>`, `<CAPTION>`, and `<MORE_DETAILED_CAPTION>` tasks) for object detection, scene captioning, visual category matching, and severity estimation |
| 🛡️ **Cross-Modal Reconciliation** | Automated mismatch checking. Disagreements between text and visual classification trigger category conflict flags and downgrade trust tiering to protect against silent categorization hijacking |
| 📹 **Desktop Webcam** | Citizens can now capture photos directly from their desktop via `getUserMedia` |
| 🤖 **AI Pipeline** | Whisper STT → IndicTrans2/NLLB translation → TF-IDF + NB (98.99% acc.) → spaCy NER → Florence-2 |
| 🛡️ **Trust Tiers** | High trust (auto-verified) for valid photo+location; Medium trust for text-only; Downgraded to `manual_review` on mismatch |
| 🗺️ **Interactive Maps** | Map widget in citizen portal for tagging; Dashboard map showing all active complaints (color-coded by status, bypasses pagination limits, and automatically removes resolved reports) |
| 👥 **Duplicate Voting** | Duplicate complaints (same category within 0.5 km and 180 days) are merged: files deleted, vote count incremented |
| 🔐 **JWT Auth** | Secure database-backed login with token-based access control |
| ✅ **HITL Verification** | Admin verifies/edits AI-classified complaints before finalizing |
| 🔊 **Audio Playback** | Officers listen to original voice recordings in the dashboard |
| 📊 **Live Statistics** | Real-time stats with category, language, and pothole severity distributions |
| 📈 **NLP Analytics Dashboard** | Comprehensive NLP metrics, energy monitoring, classifier confidence, NER quality, stage bottleneck analysis, and throughput tracking — all from real runtime data |

---

## 📁 Project Structure

```
Civic Complaint/
├── main.py                  # FastAPI backend (all endpoints)
├── model_bbmp.pkl           # Trained TF-IDF + NB classifier
├── requirements.txt         # Python dependencies
├── Dockerfile               # Backend container image
├── docker-compose.yml       # PostgreSQL + backend orchestration
├── .env.example             # Safe env template committed to git
├── .env                     # Local secrets (ignored; never commit)
│
├── scripts/
│   ├── train_bbmp_model.py  # Model training script (BBMP dataset)
│   └── insert_mock.py       # Insert mock complaints for testing
│
├── tests/
│   ├── test_backend.py      # Backend integration tests
│   ├── test_cors.py         # CORS configuration tests
│   ├── test_submit.py       # Complaint submission tests
│   └── test_whisper.py      # Whisper transcription tests
│
├── data/
│   ├── BBMP_Grievances_2023.csv   # Raw BBMP dataset (207k records)
│   ├── BBMP_cleaned.csv           # Cleaned training data
│   └── dataAnalysis.py            # Dataset analysis script
│
├── uploads/                 # Audio files (gitignored)
│
└── civic-frontend/          # React + Vite frontend
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx         # React entry point
        ├── App.jsx          # Root component + routing
        ├── components/
        │   ├── Navbar.jsx                # Sticky navigation bar
        │   ├── RecordComplaint.jsx       # Citizen voice capture page
        │   ├── ComplaintList.jsx         # Admin dashboard page
        │   └── AnalyticsDashboard.jsx    # NLP analytics & energy monitoring
        ├── styles/
        │   ├── index.css                 # Global design system (tokens)
        │   ├── Navbar.css                # Navbar styles
        │   ├── RecordComplaint.css       # Citizen portal styles
        │   ├── ComplaintList.css         # Admin dashboard styles
        │   └── AnalyticsDashboard.css    # Analytics dashboard styles
        ├── services/
        │   └── api.js                    # API client (fetch-based)
        └── assets/                       # Static assets
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm`
- **FFmpeg** (for Whisper audio processing)
- **Device location services** enabled in browser for complaint submission
- **PostgreSQL 14+** *(optional — falls back to SQLite)*

### 1. Clone & Setup Backend

```bash
git clone https://github.com/your-username/civic-complaint.git
cd civic-complaint

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Create local environment file (contains secrets; do not commit)
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# You MUST edit .env and set strong SECRET_KEY and DB_PASSWORD.
# Example of what your .env should look like:
#
# # App secrets
# SECRET_KEY=<your-very-long-random-secret>
# JWT_ALGORITHM=HS256
# ADMIN_USERNAME=admin
# ADMIN_PASSWORD=<your-secure-admin-password>
#
# # Threshold tuning parameters
# ZERO_SHOT_MIN_CONFIDENCE=0.85
# IMAGE_RECONCILE_CONFIDENCE_THRESHOLD=0.6
#
# # Database settings
# DB_USER=postgres
# DB_PASSWORD=<your-secure-db-password>
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=bbmp_complaints

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# (Optional) Retrain the TF-IDF + Naive Bayes category classifier
python scripts/train_bbmp_model.py

# If you add or upgrade Python packages, refresh the lock file before commit
pip freeze > requirements.txt

# Start the backend (recommended)
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

> Backend runs at `http://localhost:8000`
>
> Health check: `http://localhost:8000/health`

### 2. Setup Frontend

```bash
cd civic-frontend
npm install
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
npm run dev
```

> Frontend runs at `http://localhost:5173`

### 3. Database (Optional — PostgreSQL)

```bash
docker-compose up -d db
```

> Without PostgreSQL, the backend automatically falls back to SQLite (`complaints.db`).

---

## 🔑 Default Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | `ADMIN_USERNAME` from `.env` | `ADMIN_PASSWORD` from `.env` |

> ⚠️ **Note:** The backend strictly reads these credentials (as well as `SECRET_KEY` and `DB_PASSWORD`) from your `.env` file. Do not hardcode them in any code files.

---

## 🔒 Security and Secret Management

- Keep API keys, JWT secrets, and DB passwords only in local `.env` files.
- Never commit `.env`, `.env.*`, private keys, or credential JSON files.
- Use `.env.example` and `civic-frontend/.env.example` as templates with placeholder values.
- `.gitignore` blocks common private/runtime artifacts (`uploads/`, local DB files, logs, model/output folders, and virtual envs).
- If a secret is accidentally committed, rotate it immediately and remove it from git history before sharing.

### Safe Push Checklist

Run this before every `git push`:

```bash
git status --short
git ls-files .env .env.* "civic-frontend/.env*" complaints.db bbmp_complaints.log uploads/*
```

Expected result: only template files such as `.env.example` should appear.

---

## 🔐 Authenticity Rules

- Live location is mandatory for all complaint submissions.
- Submission without live location is rejected.
- Image evidence is optional, but when provided it must pass backend authenticity checks:
    - EXIF GPS metadata required.
    - EXIF timestamp required and must be at most 10 minutes old.
    - EXIF coordinates must match live location within a 100-meter radius.

Trust policy:

- Image + Live Location (EXIF match) → **High trust** (`status="verified"`, `trust_level="high"`)
- Text/Audio + Live Location → **Medium trust** (`status="pending"`, `trust_level="medium"`)
- Image/Text Category Disagreement (Cross-modal mismatch) → **Downgraded trust** (`status="pending"`, `trust_level="manual_review"`, `category_mismatch=True`)
- No Live Location → **Rejected** (HTTP 400)

---

## 👥 Duplicate Detection & Voting System

To prevent spam and keep the admin dashboard clean, the system automatically checks for duplicate complaints:
- **Criteria**: Same category, submitted within a **0.5 km radius** and **180 days** of an existing open complaint.
- **Handling**:
  - Instead of creating a new database record, the system increments the vote count of the existing complaint.
  - Unique voter fingerprints (`voter_fingerprint`) are tracked in `complaint_votes` to prevent duplicate voting by the same user.
  - Uploaded audio/image files for duplicate submissions are automatically deleted from storage to save disk space.
- **Ranking**: Complaints are sorted by priority: `votes DESC, created_at DESC` (more votes first, then newer complaints).

---

## 🧠 NLP Pipeline & Image Analysis Chain

```
🎤 Voice Input (Native Kannada/Hindi) OR Text
    ↓
📝 Whisper STT (Transcription in source language)
    ↓
🌐 IndicTrans2/NLLB (Dedicated translation step)
    ↓
🏷️ TF-IDF + Naive Bayes Classifier (98.99% accuracy)
    ↓
📸 YOLOv8n-seg Image Analysis (Multi-class visual parsing)
    ↓
⚖️ Cross-Modal Reconciliation (Checks visual category against text category)
    ↓
💾 PostgreSQL / SQLite storage (with Live GPS, Trust Scoring & Mismatch Flags)
```

---

## 🛠️ API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | ❌ | Service liveness/readiness probe |
| `POST` | `/login` | ❌ | Admin login → JWT token |
| `POST` | `/submit-complaint` | ❌ | Submit complaint with live location, optional audio/text, optional image evidence |
| `GET` | `/complaints` | 🔐 | Paginated complaint list |
| `GET` | `/complaints/stats` | 🔐 | Category & language statistics |
| `GET` | `/analytics/dashboard` | 🔐 | Full NLP analytics & energy monitoring dashboard (supports `?start_date`, `?end_date`, `?language` filters) |
| `PUT` | `/complaints/{id}/verify` | 🔐 | HITL verify/edit complaint |
| `GET` | `/uploads/{filename}` | 🔐 | Serve protected uploaded media (audio/image) |

### `POST /submit-complaint` Form Fields

- `live_latitude` (required)
- `live_longitude` (required)
- `live_location_timestamp` (required, ISO-8601)
- `file` (optional audio file)
- `text_note` (optional text complaint)
- `image` (optional evidence image)

At least one of `file` or `text_note` is required.

---

## 📊 NLP Analytics & Energy Monitoring Dashboard

The system includes a comprehensive analytics dashboard at `/analytics` (JWT-protected) that provides deep insights into NLP pipeline performance, energy consumption, and system health. **Every metric is derived from real runtime data — zero hardcoded, estimated, random, static, or sample values.**

### Key Metric Cards

| Metric | Data Source | Calculation |
|--------|------------|-------------|
| Total Complaints Processed | `nlp_metrics` `COUNT(*)` | Direct DB query |
| Unique Complaints | `complaints` `COUNT(*)` | Direct DB query |
| Duplicate Complaints | `nlp_metrics` `WHERE is_duplicate=True` | Boolean flag set by duplicate detection |
| Total Votes | `complaints` `SUM(votes)` | Direct DB aggregate |
| Avg Processing Time | `nlp_metrics` `AVG(total_processing_time)` | Measured via `time.perf_counter()` |
| Total Energy (J) | `nlp_metrics` `SUM(total_energy_joules)` | CPU TDP × measured processing time |
| Classifier Confidence | `nlp_metrics.classifier_confidence` | `sklearn predict_proba()` per request |
| Entity Count | `nlp_metrics.entity_count` | `len(spacy_doc.ents)` per request |
| Audio Duration | `nlp_metrics.audio_duration_seconds` | `pydub AudioSegment.duration_seconds` |
| Zero-shot Rate | `nlp_metrics.zero_shot_triggered` | Boolean flag per request |
| Error Rate | `nlp_metrics.error_stage` | Exception handler captures stage name |

### Charts (15+)

- **Energy by Stage** — bar chart showing energy consumption per NLP stage
- **Energy Over Time** — line chart with dual axis (Joules + request count)
- **Stage Bottleneck Radar** — radar chart identifying the slowest NLP stages
- **Throughput Over Time** — line chart showing complaints processed per day
- **Category Distribution** — doughnut chart of complaint categories
- **Source Language Distribution** — bar chart with avg processing time overlay
- **Classifier Confidence Histogram** — color-coded (red < 0.5, yellow 0.5–0.85, green > 0.85)
- **Category × Language Heatmap** — CSS-based matrix showing cross-tabulation
- **NER Entity Count Distribution** — histogram of entities extracted per complaint
- **Entity Type Breakdown** — doughnut chart (GPE vs LOC vs FAC vs ORG)
- **Audio Duration vs Processing Time** — scatter plot validating linear scaling
- **Error Rate by Stage** — bar chart identifying the most failure-prone pipeline steps
- **Duplicate vs Unique** — pie chart
- **Zero-shot Fallback Rate** — doughnut chart
- **Votes per Complaint** — bar chart (top 20)
- **Duplicate Cluster Sizes** — bar chart of vote distribution

### Energy Calculation Methodology

```
Energy (J) = Estimated CPU TDP (W) × Measured Processing Time (s)
```

- **CPU TDP** is derived from actual hardware info at startup using `platform.processor()`, `platform.machine()`, and `os.cpu_count()`.
- The CPU model string is classified into power tiers (ARM 10W, Mobile 15W, High-perf laptop 45W, Desktop 65W, Server 95W).
- **Processing time** for each NLP stage (transcription, translation, classification, NER, zero-shot) is measured with `time.perf_counter()`.
- The detection method and estimated wattage are logged at startup and stored in every `NlpMetric` record.
- The exact derivation is visible in the API response (`data_sources` field) and in the dashboard's Verification Panel.

> ⚠️ This is an estimation based on CPU TDP, not a direct hardware power measurement. The methodology is fully documented in every API response and the dashboard UI.

### Filters

The dashboard supports optional query parameters:
- `start_date` / `end_date` — ISO date range filter
- `language` — filter by source language (`en`, `kn`, `hi`)

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Vite 8, Leaflet, React Router, Chart.js |
| **Backend** | FastAPI, SQLAlchemy, Pydantic |
| **AI/ML** | OpenAI Whisper, scikit-learn, spaCy |
| **Translation** | Hugging Face Transformers (IndicTrans2 primary, NLLB fallback) |
| **Auth** | python-jose (JWT) |
| **Database** | PostgreSQL (primary), SQLite (fallback) |
| **DevOps** | Docker, Docker Compose |

---

## 🩺 Troubleshooting

### Frontend says "Server error 500"

- Confirm backend is running at `http://localhost:8000/health`.
- Check backend logs (`bbmp_complaints.log`) for the exact exception.
- For image submissions, ensure the photo contains valid EXIF GPS + timestamp metadata.
- If EXIF is missing/invalid or location mismatch exceeds 100 meters, backend returns a clear `400` error by design.

### Image submission rejected

- Capture a fresh photo from the in-app camera when possible.
- Keep GPS/location services enabled on the device/browser.
- Ensure image capture and complaint submission happen within 10 minutes.

---

## 📄 License

This project is developed for academic purposes as part of the BBMP Civic Grievance initiative, Bengaluru.
