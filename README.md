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
| 📍 **Live Location Capture** | Complaint submission requires real-time GPS coordinates from the device |
| 📷 **Image Authenticity** | Camera/gallery image evidence validated via EXIF GPS + timestamp checks |
| 📹 **Desktop Webcam** | Citizens can now capture photos directly from their desktop via `getUserMedia` |
| 🤖 **AI Pipeline** | Whisper STT (transcribe) → IndicTrans2/NLLB translation → TF-IDF + NB (98.5% acc.) → spaCy NER |
| 🛡️ **Trust Tiers** | High trust (auto-verified) for valid photo+location; Medium trust for others |
| 🗺️ **Interactive Map** | Leaflet map with **dynamic red markers** for pending complaints (removed on verify) |
| 🔐 **JWT Auth** | Secure database-backed login with token-based access control |
| ✅ **HITL Verification** | Admin verifies/edits AI-classified complaints before finalizing |
| 🔊 **Audio Playback** | Officers listen to original voice recordings in the dashboard |
| 📊 **Live Statistics** | Real-time stats with category and language breakdowns |

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
        │   ├── Navbar.jsx          # Sticky navigation bar
        │   ├── RecordComplaint.jsx # Citizen voice capture page
        │   └── ComplaintList.jsx   # Admin dashboard page
        ├── styles/
        │   ├── index.css           # Global design system (tokens)
        │   ├── Navbar.css          # Navbar styles
        │   ├── RecordComplaint.css # Citizen portal styles
        │   └── ComplaintList.css   # Admin dashboard styles
        ├── services/
        │   └── api.js              # API client (fetch-based)
        └── assets/                 # Static assets
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

# Edit .env and set strong SECRET_KEY / DB credentials

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

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
| Admin | `ADMIN_USERNAME` (default: `admin`) | `ADMIN_PASSWORD` from `.env` |

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

- Image + Live Location → **High trust** (`auto_verified`)
- Text/Audio + Live Location → **Medium trust** (`manual_review`)
- No Live Location → **Rejected**

---

## 🧠 NLP Pipeline

```
🎤 Voice Input (Native Kannada)
    ↓
📝 Whisper STT (Transcription in source language)
    ↓
🌐 IndicTrans2/NLLB (Dedicated translation step)
    ↓
🏷️ TF-IDF + Naive Bayes Classifier (98.5% accuracy)
    ↓
💾 PostgreSQL / SQLite storage (with Live GPS)
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

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Vite 8, Leaflet, React Router |
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
