<div align="center">

# 🏛️ BBMP Civic Complaint System

**A Multilingual, AI-Powered Voice Complaint Platform for Bengaluru Citizens**

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
| 🤖 **AI Pipeline** | Whisper STT → Google Translate → TF-IDF + Naive Bayes classifier (98.5% acc.) → spaCy NER geo-tagging |
| 🗺️ **Interactive Map** | Leaflet + CartoDB Positron light tiles with color-coded complaint markers |
| 🔐 **JWT Auth** | Secure admin login with token-based access control |
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
├── .env                     # Environment variables (secrets)
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
- **PostgreSQL 14+** *(optional — falls back to SQLite)*

### 1. Clone & Setup Backend

```bash
git clone https://github.com/your-username/civic-complaint.git
cd civic-complaint

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start the backend
python main.py
```

> Backend runs at `http://localhost:8000`

### 2. Setup Frontend

```bash
cd civic-frontend
npm install
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
| Admin | `admin` | `bbmp2025` |

---

## 🧠 NLP Pipeline

```
🎤 Voice Input (Kannada/Hindi/English)
    ↓
📝 Whisper STT (speech-to-text)
    ↓
🌐 Google Translate → English
    ↓
🏷️ TF-IDF + Naive Bayes Classifier (98.5% accuracy)
    ↓
📍 spaCy NER (GPE/LOC/FAC entity extraction)
    ↓
💾 PostgreSQL / SQLite storage
```

---

## 🛠️ API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/login` | ❌ | Admin login → JWT token |
| `POST` | `/submit-complaint` | ❌ | Upload audio → full NLP pipeline |
| `GET` | `/complaints` | 🔐 | Paginated complaint list |
| `GET` | `/complaints/stats` | 🔐 | Category & language statistics |
| `PUT` | `/complaints/{id}/verify` | 🔐 | HITL verify/edit complaint |
| `GET` | `/uploads/{filename}` | 🔐 | Serve original audio file |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Vite 8, Leaflet, React Router |
| **Backend** | FastAPI, SQLAlchemy, Pydantic |
| **AI/ML** | OpenAI Whisper, scikit-learn, spaCy |
| **Translation** | deep-translator (Google Translate) |
| **Auth** | python-jose (JWT) |
| **Database** | PostgreSQL (primary), SQLite (fallback) |
| **DevOps** | Docker, Docker Compose |

---

## 📄 License

This project is developed for academic purposes as part of the BBMP Civic Grievance initiative, Bengaluru.
