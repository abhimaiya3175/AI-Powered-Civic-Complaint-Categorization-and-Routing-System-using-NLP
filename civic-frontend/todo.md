Here's the full task list as plain text — just select all and copy!

```
BBMP Civic Complaint System — Project Task List
NLP Project · AI363IA · Abhilash Maiya, Abhishek Biradar, Harsh Agarwal

================================================================
PHASE 0 — FOUNDATION ✅ DONE
================================================================
[x] BBMP 2024 dataset loaded (207k complaints)
[x] TF-IDF + Naive Bayes model trained & saved (model_bbmp.pkl — 98.5% accuracy)
[x] FastAPI backend created (main.py with Whisper + spaCy + SQLite)

================================================================
PHASE 1 — BACKEND FINAL SETUP ✅ DONE
================================================================
[x] Create uploads/ and frontend/ folders
[x] Run python main.py and verify Uvicorn starts on port 8000
[x] Record a 5-second Kannada/English .wav and test POST /submit-complaint via curl
[x] Confirm complaint appears in complaints.db via DB Browser for SQLite
[x] Upgrade database to PostgreSQL (synopsis requires relational DB)
[x] Store audio file path alongside each complaint record in the database

================================================================
PHASE 2 — TRANSLATION PIPELINE ✅ DONE
================================================================
[x] Integrate Translation API: Kannada/Hindi STT output → English text
[x] Verify full pipeline: voice → STT (Whisper) → translate → NLP classify → NER geo-tag

================================================================
PHASE 3 — CITIZEN PORTAL / REACT FRONTEND ✅ DONE
================================================================
[x] Scaffold React app (Vite)
[x] Install dependencies: react-router-dom, leaflet, react-leaflet
[x] Build RecordComplaint.jsx — voice capture, submit, show predicted category & location
[x] Build App.jsx with routing for / and /admin
[x] Run npm run dev and confirm frontend works

================================================================
PHASE 4 — ADMIN DASHBOARD + JWT AUTH + HITL ✅ DONE
================================================================
[x] Build ComplaintList.jsx — table of complaints with status column
[x] JWT-based secure authentication (python-jose, OAuth2)
[x] Implement HITL flow: submit → admin sees pending → Verify/Edit → status = Verified
[x] Add audio playback in admin table so officers can listen to original voice recording

================================================================
PHASE 5 — TESTING & POLISH ✅ DONE
================================================================
[x] Multilingual testing — test files in tests/ directory
[x] Backend tests: test_backend.py, test_cors.py, test_submit.py, test_whisper.py
[x] Add Leaflet + OpenStreetMap map showing extracted geo-tag locations in admin dashboard

================================================================
PHASE 6 — DEPLOYMENT ✅ DONE
================================================================
[x] Create Dockerfile and docker-compose.yml
[ ] Deploy backend to Render.com (free tier) or Railway.app
[ ] Deploy React frontend to Vercel
[x] Push to public GitHub repo with .gitignore and README
================================================================
PHASE 7 — OPTIONAL ENHANCEMENTS
================================================================
[ ] Improve "Others" class with more training data
[ ] Add email/SMS notification to ward officer on complaint verification
[ ] Add photo upload alongside voice complaint
[ ] Fine-tune Whisper for Bengaluru Kannada accent (IndicTTS dataset)
[ ] Build mobile PWA or React Native version
```