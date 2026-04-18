BBMP VOICE COMPLAINT SYSTEM — PROJECT TO-DO
Bengaluru · Multilingual + HITL · NLP Project
=============================================

PHASE 0 — FOUNDATION ✅ DONE
-------------------------------
[x] BBMP 2024 dataset loaded (207k complaints)
[x] TF-IDF + Naive Bayes model trained & saved (model_bbmp.pkl — 98.5% accuracy)
[x] FastAPI backend created (main.py with Whisper + spaCy + SQLite)

PHASE 1 — BACKEND FINAL SETUP ✅ DONE
------------------------------------------------
[x] Create uploads/ and frontend/ folders
[x] Run python main.py and verify Uvicorn starts on port 8000
[x] Record a 5-second Kannada/English .wav and test POST /submit-complaint via curl
[x] Confirm complaint appears in complaints.db via DB Browser for SQLite
[x] ⚠️ NEW — Upgrade database to PostgreSQL  (synopsis requires relational DB)
[x] ⚠️ NEW — Store audio file path alongside each complaint record in the database

PHASE 2 — TRANSLATION PIPELINE ✅ DONE
-------------------------------------------------
[x] ⚠️ NEW — Integrate Translation API: Kannada/Hindi STT output → English text
[x] Verify full pipeline: voice → STT (Whisper) → translate → NLP classify → NER geo-tag

PHASE 3 — CITIZEN PORTAL / REACT FRONTEND ✅ DONE
----------------------------------------------------------
[x] Scaffold React TypeScript app: npx create-react-app . --template typescript
[x] Install dependencies: axios, @mui/material, @emotion/react, @emotion/styled, react-router-dom
[x] Build RecordComplaint.tsx — voice capture, submit, show predicted category & location
[x] Build App.tsx with routing for /citizen and /admin
[x] Run npm start and confirm http://localhost:3000 works end-to-end

PHASE 4 — ADMIN DASHBOARD + JWT AUTH + HITL ✅ DONE
--------------------------------------------------------------
[x] Build AdminDashboard.tsx — table of pending complaints with status column
[x] Replace hardcoded password with JWT-based secure authentication (from synopsis)
[x] Implement HITL flow: submit → admin sees pending → Verify/Edit → status = Verified
[x] Add audio playback in admin table so officers can listen to original voice recording

PHASE 5 — TESTING & POLISH ✅ DONE
-------------------------------------------
[x] Multilingual testing — record 5 Kannada complaints (IndicTTS samples)
[x] Record 5 Hindi complaints and check translation + classification accuracy
[x] Record 5 English complaints and verify end-to-end in admin dashboard
[x] Add Leaflet + OpenStreetMap map showing extracted geo-tag locations in admin dashboard

PHASE 6 — DEPLOYMENT (Day 3 · 1 hr)
--------------------------------------
[x] Create Dockerfile and docker-compose.yml
[x] Deploy backend to Render.com (free tier) or Railway.app
[x] Deploy React frontend to Vercel
[x] Push to public GitHub repo with .gitignore and README with screenshots

PHASE 7 — OPTIONAL ENHANCEMENTS
----------------------------------
[ ] Improve "Others" class with more training data
[ ] Add email/SMS notification to ward officer on complaint verification
[x] Add photo upload alongside voice complaint
[ ] Fine-tune Whisper for Bengaluru Kannada accent (IndicTTS dataset)
[ ] Build mobile PWA or React Native version

=============================================
⚠️ NEW = Added from NLP Synopsis (missing from chat)