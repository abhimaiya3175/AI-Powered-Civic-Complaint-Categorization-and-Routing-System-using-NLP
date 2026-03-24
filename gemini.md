BBMP VOICE COMPLAINT SYSTEM — PROJECT TO-DO
Bengaluru · Multilingual + HITL · NLP Project
=============================================

PHASE 0 — FOUNDATION ✅ DONE
-------------------------------
[x] BBMP 2024 dataset loaded (207k complaints)
[x] TF-IDF + Naive Bayes model trained & saved (model_bbmp.pkl — 98.5% accuracy)
[x] FastAPI backend created (main.py with Whisper + spaCy + SQLite)

PHASE 1 — BACKEND FINAL SETUP (Today · 30 min)
------------------------------------------------
[ ] Create uploads/ and frontend/ folders
[ ] Run python main.py and verify Uvicorn starts on port 8000
[ ] Record a 5-second Kannada/English .wav and test POST /submit-complaint via curl
[ ] Confirm complaint appears in complaints.db via DB Browser for SQLite
[ ] ⚠️ NEW — Upgrade database to PostgreSQL  (synopsis requires relational DB)
[ ] ⚠️ NEW — Store audio file path alongside each complaint record in the database

PHASE 2 — TRANSLATION PIPELINE (Today · 30 min)
-------------------------------------------------
[ ] ⚠️ NEW — Integrate Translation API: Kannada/Hindi STT output → English text
[ ] Verify full pipeline: voice → STT (Whisper) → translate → NLP classify → NER geo-tag

PHASE 3 — CITIZEN PORTAL / REACT FRONTEND (Today · 1 hr)
----------------------------------------------------------
[ ] Scaffold React TypeScript app: npx create-react-app . --template typescript
[ ] Install dependencies: axios, @mui/material, @emotion/react, @emotion/styled, react-router-dom
[ ] Build RecordComplaint.tsx — voice capture, submit, show predicted category & location
[ ] Build App.tsx with routing for /citizen and /admin
[ ] Run npm start and confirm http://localhost:3000 works end-to-end

PHASE 4 — ADMIN DASHBOARD + JWT AUTH + HITL (Day 2 · 45 min)
--------------------------------------------------------------
[ ] Build AdminDashboard.tsx — table of pending complaints with status column
[ ] Replace hardcoded password with JWT-based secure authentication (from synopsis)
[ ] Implement HITL flow: submit → admin sees pending → Verify/Edit → status = Verified
[ ] Add audio playback in admin table so officers can listen to original voice recording

PHASE 5 — TESTING & POLISH (Day 2 · 1 hr)
-------------------------------------------
[ ] Multilingual testing — record 5 Kannada complaints (IndicTTS samples)
[ ] Record 5 Hindi complaints and check translation + classification accuracy
[ ] Record 5 English complaints and verify end-to-end in admin dashboard
[ ] Add Leaflet + OpenStreetMap map showing extracted geo-tag locations in admin dashboard

PHASE 6 — DEPLOYMENT (Day 3 · 1 hr)
--------------------------------------
[ ] Create Dockerfile and docker-compose.yml
[ ] Deploy backend to Render.com (free tier) or Railway.app
[ ] Deploy React frontend to Vercel
[ ] Push to public GitHub repo with .gitignore and README with screenshots

PHASE 7 — OPTIONAL ENHANCEMENTS
----------------------------------
[ ] Improve "Others" class with more training data
[ ] Add email/SMS notification to ward officer on complaint verification
[ ] Add photo upload alongside voice complaint
[ ] Fine-tune Whisper for Bengaluru Kannada accent (IndicTTS dataset)
[ ] Build mobile PWA or React Native version

=============================================
⚠️ NEW = Added from NLP Synopsis (missing from chat)