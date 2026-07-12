# Technical Overview of Civic Complaint System

This document outlines the complete technical architecture, software stack, and machine learning integration of the **AI-Powered Civic Complaint Categorization and Routing System**.

## 1. System Architecture

The project follows a modern **Client-Server Architecture** augmented with an embedded **Machine Learning Pipeline**. It leverages a decoupled React frontend and a FastAPI backend, communicating via RESTful endpoints.

- **Frontend (Client Layer)**: Handles user interaction, voice recording, image uploads, metadata extraction (EXIF GPS data), and renders maps and dashboards.
- **Backend (Service Layer)**: Orchestrates the database transactions, duplicate checking, and routes data to the ML pipeline.
- **ML Pipeline (Intelligence Layer)**: Processes multimodal inputs (audio, text, image), translates local languages, extracts features, and runs inferences to predict civic categories and assess damage severity.

## 2. Technology Stack

### Frontend (User Interface)
- **Framework**: React.js (v19) powered by Vite for rapid HMR and optimized builds.
- **Routing**: React Router DOM for SPA navigation.
- **Maps**: Leaflet and React-Leaflet for interactive geographic plotting of complaints.
- **Data Visualization**: Chart.js and React-Chartjs-2 for analytics on the admin dashboard.
- **Animations & UX**: Framer Motion for micro-interactions; Vanilla CSS for rich, responsive, and dynamic styling (glassmorphism, gradient badges).
- **Utility**: Piexifjs for extracting GPS coordinates directly from uploaded images.

### Backend (API & Business Logic)
- **Framework**: FastAPI (Python), providing highly concurrent, type-safe REST APIs.
- **Server**: Uvicorn (ASGI) running on Python 3.10+.
- **Database ORM**: SQLAlchemy 2.0.
- **Database Engine**: PostgreSQL for robust relational data storage and geospatial latitude/longitude queries.
- **Environment Management**: Python `dotenv` for configuration handling.

## 3. Artificial Intelligence & Machine Learning Stack

The core innovation of this project lies in its multimodal AI pipeline, which integrates state-of-the-art open-source models:

### Text Classification & NLP
- **Model**: Custom-trained Scikit-Learn Pipeline (`model_bbmp.pkl`).
- **Algorithm**: SVM / Logistic Regression trained on a curated civic dataset of 40,000+ records.
- **Features**: TF-IDF Vectorization, mapped to 14 distinct civic categories (e.g., Road Repair, Garbage, Street Light, Water Supply).
- **Processing**: `spaCy` (`en_core_web_sm`) is used for text tokenization, lemmatization, and stop-word removal.

### Multilingual Translation
- **Model**: `ai4bharat/indictrans2-indic-en-dist-200M` (Hugging Face Transformers).
- **Purpose**: Translates vernacular complaints (Kannada, Hindi, Tamil, Telugu, etc.) into English before routing them to the core classifier. This ensures high accuracy regardless of the citizen's native language.

### Speech-to-Text (Voice Complaints)
- **Model**: OpenAI `Whisper` (Small).
- **Purpose**: Transcribes audio complaints recorded by users in the browser. It features inherent multilingual capabilities, directly outputting text that is then processed by the translation and classification pipelines.

### Vision & Image Analysis
- **Model**: `microsoft/Florence-2`.
- **Purpose**: Acts as a Vision-Language Model (VLM) for deep image understanding.
- **Capabilities**: 
  - Generates descriptive image captions.
  - Performs object detection with bounding boxes (e.g., identifying potholes, garbage piles).
  - Assesses severity levels (e.g., "Critical Damage").
  - Identifies cross-modal mismatches (e.g., user text says "Pothole" but image shows a "Broken Streetlight").

## 4. Key Technical Features

- **Cross-Modal Verification**: The system compares the output of the text classifier against the visual understanding of the Florence-2 model. If they contradict, it flags a "Mismatch" for manual administrative review.
- **Tiered Duplicate Detection**: 
  1. **Spatial Filter**: Uses Haversine distance (GPS coordinates) to find nearby complaints within a 200-meter radius.
  2. **Semantic Filter**: Uses Jaccard similarity on tokenized text to confirm if the nearby complaint is about the same issue.
- **Dynamic Vote Priority System**: The system dynamically calculates priority thresholds (Critical, High, Medium) based on the global maximum vote count, updating UI borders, badges, and map circle radii in real-time.
- **Explainable AI (XAI)**: The backend provides a `decision_path` detailing exactly why a specific category was chosen (e.g., matching specific keywords or visual cues).
