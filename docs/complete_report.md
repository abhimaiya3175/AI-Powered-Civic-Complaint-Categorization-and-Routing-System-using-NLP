# Complete Project Report: AI-Powered Civic Complaint Categorization and Routing System

## 1. Abstract
Urban infrastructure management suffers from inefficiencies due to the manual sorting of thousands of daily civic complaints. These complaints are often submitted in diverse regional languages, lack clear categorization, and frequently duplicate existing reports, leading to resource misallocation. 

This project presents a multimodal, AI-powered system designed to automate the logging, translation, categorization, and deduplication of civic issues. By integrating state-of-the-art Natural Language Processing (NLP), Speech-to-Text, and Vision-Language Models (VLMs), the system enables citizens to report issues seamlessly via voice or images. The backend intelligently categorizes complaints, identifies cross-modal mismatches, prevents duplicate entries via spatial-semantic filtering, and prioritizes critical issues using a community voting mechanism. The result is a highly efficient, scalable platform that bridges the communication gap between citizens and municipal authorities.

## 2. Objectives
1. **Automate Categorization**: Develop an ML model to automatically classify free-text and transcribed complaints into 14 predefined civic categories (e.g., Road Repair, Garbage, Water Supply).
2. **Enable Multilingual Access**: Break language barriers by allowing citizens to submit audio or text complaints in vernacular languages (e.g., Kannada, Hindi), automatically translating them to English for processing.
3. **Multimodal Verification**: Utilize a Vision-Language Model (VLM) to analyze uploaded images to extract evidence, assess severity, and verify that the image content matches the text description.
4. **Eliminate Redundancy**: Implement a robust duplicate detection system combining geolocation clustering (Haversine formula) and semantic text similarity (Jaccard index).
5. **Prioritize Community Urgency**: Convert duplicate reports into "upvotes" and implement a dynamic priority tier system to visually highlight critical issues on public and administrative dashboards.

## 3. Methodology

### 3.1 Data Collection & Preprocessing
A dataset of approximately 40,000+ historical civic complaints was utilized. The data underwent rigorous cleaning, including the removal of null values, deduplication, and text normalization using the `spaCy` library (tokenization, stop-word removal, and lemmatization).

### 3.2 NLP Classification Pipeline
A Scikit-Learn pipeline was built using TF-IDF vectorization coupled with a Logistic Regression / Support Vector Machine (SVM) classifier. The model was trained on the preprocessed text to predict 14 distinct classes. Hyperparameter tuning ensured high macro-average precision and recall across all categories.

### 3.3 Multimodal AI Integration
- **Speech-to-Text**: Integrated OpenAI's `Whisper` model to handle real-time browser audio blobs, transcribing spoken complaints into text.
- **Translation**: Employed `ai4bharat/indictrans2-indic-en-dist-200M` to detect and translate regional languages into English, ensuring the core NLP classifier operates on unified semantic data.
- **Computer Vision**: Integrated `microsoft/Florence-2` to run inferences on uploaded images. The model was prompted to act as an infrastructure inspector, returning bounding boxes, captions, and severity ratings.

### 3.4 Duplicate Detection & Voting Logic
When a new complaint is submitted, the system filters the PostgreSQL database for existing active complaints within a 200-meter radius using the Haversine formula based on EXIF-extracted GPS metadata. It then calculates the Jaccard similarity between the new and existing tokenized texts. If the similarity exceeds a predefined threshold (e.g., 0.35), the new submission is discarded, and the existing complaint's vote count is incremented by 1.

### 3.5 System Architecture & User Interface
The system was developed with a decoupled architecture:
- **Backend**: FastAPI orchestrates the ML models, handles REST API requests, and manages the PostgreSQL database via SQLAlchemy.
- **Frontend**: React (Vite) provides two primary interfaces. The Public UI allows citizens to view a live interactive map (React-Leaflet) and submit complaints. The Admin UI provides a dashboard for officials to review AI inferences (explainability paths, bounding box overlays) and update complaint statuses.

## 4. Results

1. **High Classification Accuracy**: The custom NLP classifier achieved an impressive overall accuracy of **99.05%**, with macro averages for precision, recall, and f1-score all hitting 0.98. The model accurately discerns subtle differences between categories like "Parks" and "Parks / Forest".
2. **Successful Cross-Modal Flagging**: The integration of Florence-2 successfully flags contradictory submissions. For example, if a user submits a picture of a tree but verbally complains about a pothole, the system correctly tags the submission with a `Mismatch Warning` for manual review.
3. **Effective Deduplication**: The spatial-semantic filter effectively curtails duplicate spam. Identical issues reported by different users in close proximity successfully merge into a single, high-priority ticket with an aggregated vote count.
4. **Enhanced UI Visibility**: The dynamic voting system successfully prioritizes complaints. The front-end renders distinct visual cues (pulsing red borders and badges for Critical issues, orange for High priority) based on relative vote thresholds, directly guiding administrative attention.

## 5. Conclusion
The AI-Powered Civic Complaint Categorization and Routing System successfully demonstrates how modern AI can modernize urban governance. By accommodating multilingual input and multimodal evidence, the platform ensures inclusivity for all citizens. Furthermore, the automated deduplication and AI-driven priority sorting significantly reduce the administrative overhead required to triage complaints. Future enhancements could include predictive analytics for infrastructure failure and direct integration into municipal ticketing systems (e.g., JIRA).
