# 🧠 Technical Architecture — BBMP Civic Complaint System

> A deep-dive into the NLP pipeline, ML models, and system architecture that power the multilingual AI-driven civic complaint platform.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [NLP Pipeline — End-to-End](#2-nlp-pipeline--end-to-end)
3. [Stage 1 · Speech-to-Text (Whisper STT)](#3-stage-1--speech-to-text-whisper-stt)
4. [Stage 2 · Machine Translation (IndicTrans2 / NLLB)](#4-stage-2--machine-translation-indictrans2--nllb)
5. [Stage 3 · Text Classification (TF-IDF + Multinomial Naive Bayes)](#5-stage-3--text-classification-tf-idf--multinomial-naive-bayes)
6. [Stage 4 · Named Entity Recognition (spaCy NER)](#6-stage-4--named-entity-recognition-spacy-ner)
7. [Stage 5 · Zero-Shot Semantic Fallback](#7-stage-5--zero-shot-semantic-fallback)
8. [Multilingual Feature Engineering](#8-multilingual-feature-engineering)
9. [Civic Translation Glossary](#9-civic-translation-glossary)
10. [Model Training Details](#10-model-training-details)
11. [NLP Metrics & Energy Monitoring](#12-nlp-metrics--energy-monitoring)
12. [Backend Architecture](#13-backend-architecture)
13. [Frontend Architecture](#14-frontend-architecture)
14. [Database Schema](#14-database-schema)
15. [Deployment & DevOps](#15-deployment--devops)
16. [Stage 6 · Image Analysis & Reconciliation (YOLOv8n-seg)](#16-stage-6--image-analysis--reconciliation-yolov8n-seg)

---

## 1. System Overview

The BBMP Civic Complaint System is an **end-to-end multilingual AI platform** that accepts citizen complaints in **Kannada, Hindi, or English** via voice or text, and automatically routes them to the correct civic department using a chained NLP pipeline.

```
Voice / Text Input
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                     NLP PIPELINE                            │
│                                                             │
│  [1] Whisper STT ──► [2] IndicTrans2 ──► [3] TF-IDF + NB  │
│                                      └──► [4] spaCy NER    │
│                                      └──► [5] Zero-Shot    │
│  [6] Image Analysis (YOLOv8n-seg) ◄────────────────────────┘
│    └──► Cross-Modal Category Reconciliation & Trust Grading
└─────────────────────────────────────────────────────────────┘
      │
      ▼
  PostgreSQL / SQLite + GPS + Trust Scoring + Mismatch Flag + Duplicate Detection
```

**Core NLP Technologies:**

| Library | Version | Role |
|---------|---------|------|
| `openai-whisper` | 20250625 | Speech-to-text transcription |
| `transformers` | 4.57.6 | IndicTrans2 & NLLB translation models |
| `sentencepiece` | 0.2.1 | Subword tokenisation for translation |
| `scikit-learn` | 1.7.2 | TF-IDF vectorisation + Naive Bayes classifier |
| `spacy` + `en_core_web_sm` | 3.8.11 / 3.8.0 | Named Entity Recognition |
| `torch` | 2.10.0+cpu | Deep learning inference runtime |
| `pydub` | 0.25.1 | Audio duration measurement |

---

## 2. NLP Pipeline — End-to-End

Each complaint traverses up to **five sequential NLP stages**. Processing time for every stage is measured with `time.perf_counter()` and persisted to the `nlp_metrics` table for analytics.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Speech-to-Text (Whisper "small", multilingual)                │
│  Input : audio/webm or audio/ogg blob                                    │
│  Output: raw transcript in source language (kn / hi / en)                │
├──────────────────────────────────────────────────────────────────────────┤
│  STAGE 2 — Translation (IndicTrans2 primary, NLLB-600M fallback)         │
│  Input : source-language transcript or raw text                          │
│  Output: English translation + civic glossary correction                 │
├──────────────────────────────────────────────────────────────────────────┤
│  STAGE 3 — Classification (TF-IDF FeatureUnion + MultinomialNB)          │
│  Input : build_multilingual_classification_text(english, original)       │
│  Output: predicted category + predict_proba confidence score             │
├──────────────────────────────────────────────────────────────────────────┤
│  STAGE 4 — NER (spaCy en_core_web_sm)                                    │
│  Input : English translation                                             │
│  Output: geographic entities (GPE, LOC, FAC, ORG) + entity count        │
├──────────────────────────────────────────────────────────────────────────┤
│  STAGE 5 — Zero-Shot Fallback (distilbart-mnli, optional)               │
│  Triggered when: classifier confidence < ZERO_SHOT_MIN_CONFIDENCE (0.85) │
│  Output: higher-confidence category override if score ≥ 0.55            │
├──────────────────────────────────────────────────────────────────────────┤
│  STAGE 6 — Image Analysis (YOLOv8n-seg "civic_multiclass_seg_best.pt")   │
│  Input : evidence image file                                             │
│  Output: class labels, confidence, bbox coordinates, segment polygons   │
│  Reconciliation: flags mismatches against text classification and        │
│                 downgrades trust tier to manual_review if discrepant    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1 · Speech-to-Text (Whisper STT)

### Model
- **Model**: OpenAI Whisper `small` (244M parameters, multilingual)
- **Loaded at startup** via `whisper.load_model("small")`
- **Inference**: runs on CPU using PyTorch 2.10.0+cpu

### Input Handling
- Accepts `audio/webm`, `audio/ogg`, `audio/wav`, and other formats supported by FFmpeg
- Audio duration is measured before transcription using `pydub.AudioSegment.duration_seconds`

### Transcription Behaviour
- Whisper automatically **detects the source language** (Kannada `kn`, Hindi `hi`, English `en`)
- The raw transcript is preserved in the database as `original_text` (source-language text)
- For English inputs, the transcription is used directly as the classification input without translation

### Key Parameters
```python
whisper_model = whisper.load_model("small")
result = whisper_model.transcribe(audio_path)
# result["text"]     → raw transcript
# result["language"] → detected language code
```

### Performance Measurement
```python
t0 = time.perf_counter()
result = await asyncio.to_thread(whisper_model.transcribe, audio_path)
transcription_time = time.perf_counter() - t0  # stored in nlp_metrics
```

---

## 4. Stage 2 · Machine Translation (IndicTrans2 / NLLB)

### Translation Model Chain

The system implements a **three-tier translation strategy** with automatic runtime fallback:

```
Tier 1  ai4bharat/indictrans2-indic-en-dist-200M   (primary, 200M params)
  ↓ fails
Tier 2  facebook/nllb-200-distilled-600M            (first fallback, 600M)
  ↓ fails
Tier 3  prajdabre/rotary-indictrans2-indic-en-dist-200M  (second fallback)
```

### IndicTrans2 — Primary Path

- **Architecture**: Seq2Seq Transformer (`AutoModelForSeq2SeqLM`)
- **Tokenizer**: `AutoTokenizer` with SentencePiece subword model
- **Preprocessing**: `IndicProcessor` from `IndicTransToolkit` — handles script-level tokenisation, byte-pair encoding for Indic scripts, and adds language-tag prefixes
- **Inference**:
  ```python
  encoded_inputs = tokenizer(
      preprocessed_batch,
      return_tensors="pt",
      padding=True,
      truncation=True,
  )
  generated_tokens = model.generate(**encoded_inputs, num_beams=4)
  decoded_batch = tokenizer.batch_decode(
      generated_tokens,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=True,
  )
  ```
- **Post-processing**: `IndicProcessor.postprocess_batch()` restores normalised Indic characters
- **Language tags**: Internal Flores-200 format (e.g., `kan_Knda → eng_Latn` for Kannada→English)

### NLLB — Fallback Path

- **Architecture**: `facebook/nllb-200-distilled-600M`, a massively multilingual Seq2Seq model
- **Language tags**: NLLB BCP-47-style (e.g., `kn → eng_Latn`)
- `forced_bos_token_id` is set to the English target token ID to enforce English output
- Uses `tokenizer.src_lang` to declare the source language

### Concurrency Control
- A single `asyncio.Lock` (`translation_lock`) serialises all translation requests to prevent VRAM/CPU contention during concurrent complaint submissions
- Translation runs in `asyncio.to_thread()` to avoid blocking the async event loop

---

## 5. Stage 3 · Text Classification (TF-IDF + Multinomial Naive Bayes)

This is the **core categorisation engine**. It achieves **98.99% accuracy** on the BBMP 2023 dataset.

### Feature Extraction — FeatureUnion

The vectoriser is a **`sklearn.pipeline.FeatureUnion`** combining two complementary TF-IDF representations:

```python
vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(
        ngram_range=(1, 2),     # unigrams + bigrams
        stop_words="english",   # remove common English stop words
        min_df=2,               # ignore very rare tokens
        max_df=0.95,            # ignore near-universal tokens
        sublinear_tf=True,      # log-normalise term frequency
    )),
    ("char_wb", TfidfVectorizer(
        analyzer="char_wb",     # character n-grams (word-boundary aware)
        ngram_range=(3, 5),     # 3-to-5 character windows
        min_df=2,
        max_df=0.98,
        max_features=12000,     # cap vocabulary size
        sublinear_tf=True,
    )),
])
```

**Why FeatureUnion?**
- `word` TF-IDF captures **lexical semantics**: civic keywords like "pothole", "garbage", "water supply"
- `char_wb` TF-IDF captures **morphological patterns**: robust to spelling variations, transliterated Kannada/Hindi words, and partial-word matches (e.g., "drain" matching "drainage", "drained")
- Character n-grams are especially effective for multilingual text where subword units carry domain signals

### TF-IDF Weighting

For each term $t$ in document $d$:

$$\text{TF-IDF}(t,d) = (1 + \log \text{tf}(t,d)) \times \log\frac{N+1}{\text{df}(t)+1}$$

- `sublinear_tf=True` applies the log-normalisation `1 + log(tf)` instead of raw `tf`
- IDF dampens common terms across the corpus; rare but distinctive civic terms receive higher weight

### Classifier — Multinomial Naive Bayes

```python
clf = MultinomialNB()
clf.fit(X_train_vec, y_train, sample_weight=sample_weights)
```

- **Class-balanced sample weights** computed via `compute_sample_weight(class_weight='balanced')` to handle imbalanced BBMP categories
- **Inference**: `clf.predict_proba(text_vector)[0]` — returns per-class probability scores
- The **predicted class** is the `argmax` of the probability vector
- The **confidence score** (highest probability) is stored in `nlp_metrics.classifier_confidence`

### Complaint Categories (16 classes)

| Category | BBMP Department |
|----------|----------------|
| Road Repair | Engineering / Roads |
| Street Light | Electrical |
| Garbage / Sanitation | Solid Waste Management |
| Water Supply | BWSSB |
| Drainage / SWD | Storm Water Drain |
| Health / Sanitation | Health Department |
| Parks | Parks & Open Spaces |
| Traffic | Traffic Engineering Cell |
| Town Planning | Town Planning |
| Revenue | Revenue Department |
| Veterinary | Veterinary |
| Advertisement | Advertisement |
| Others | General |
| Non-Civic | (filtered out) |

### Feature Importance — Explainability

At inference time, the top contributing TF-IDF terms per prediction are exposed via the `/classifier-features` endpoint:

```python
# For each non-zero feature in the input vector
for feature_idx, tfidf_value in zip(row.indices, row.data):
    weight_delta = log_prob_predicted - log_prob_second_best
    contribution = float(tfidf_value) * weight_delta
    # Returns term, tfidf score, and contribution to prediction
```

This provides **human-readable explainability** for HITL (Human-in-the-Loop) officers reviewing AI classifications.

---

## 6. Stage 4 · Named Entity Recognition (spaCy NER)

### Model
- **Model**: `en_core_web_sm` (spaCy 3.8.0 — English small pipeline, ~12MB)
- **Loaded at startup**: `nlp = spacy.load("en_core_web_sm")`
- **Fallback**: If the model is not installed, `spacy.blank("en")` is used (no NER output)

### Extracted Entity Types

| Entity Type | Meaning | Example |
|-------------|---------|---------|
| `GPE` | Geo-political entity (city, district) | "Jayanagar", "Bengaluru" |
| `LOC` | Geographic location | "Outer Ring Road" |
| `FAC` | Facility / infrastructure | "Metro Station", "Bus Stop" |
| `ORG` | Organisation | "BBMP Ward 56" |

### Usage in Pipeline

```python
doc = nlp(english_translation)
entities = [(ent.text, ent.label_) for ent in doc.ents]
entity_count = len(doc.ents)  # stored in nlp_metrics.entity_count
```

- Extracted entities are stored alongside the complaint to assist officers in **geographic routing**
- `entity_count` is tracked in the analytics dashboard with a histogram and entity-type doughnut chart

---

## 7. Stage 5 · Zero-Shot Semantic Fallback

### Purpose
When the primary TF-IDF + NB classifier returns a confidence score below the threshold (`ZERO_SHOT_MIN_CONFIDENCE = 0.85`), a **zero-shot MNLI model** provides a semantic second opinion.

### Model
```
valhalla/distilbart-mnli-12-1
```
A distilled BART model fine-tuned on MultiNLI — supports **zero-shot text classification** via natural language hypothesis entailment.

### Mechanism

```python
candidate_labels = list(category_map.values())  # 14 civic categories
result = zero_shot_classifier(text, candidate_labels, multi_label=False)
# result["scores"][0] → best semantic match score
# result["labels"][0] → predicted category
```

**Fallback activation conditions:**
1. Primary confidence < `ZERO_SHOT_MIN_CONFIDENCE` (0.85)
2. Zero-shot score ≥ `ZERO_SHOT_MIN_SCORE` (0.55)
3. For very sparse inputs (few tokens): zero-shot score ≥ `ZERO_SHOT_SPARSE_MIN_SCORE` (0.60)

### Lazy Loading
- The zero-shot model is **loaded on-demand** (first low-confidence prediction), not at startup, to conserve memory
- A `threading.Lock` prevents concurrent initialisation race conditions

### Metrics Tracked
- `zero_shot_triggered` (Boolean) — whether fallback was invoked
- `zero_shot_confidence` (Float) — the MNLI entailment score
- `zero_shot_time` (Float) — inference duration in seconds

---

## 8. Multilingual Feature Engineering

### Classification Text Construction

The key insight for multilingual classification is combining **English translation** and **original source-language text** into a single feature string:

```python
# nlp_features.py
def build_multilingual_classification_text(
    original_text: str,
    translated_english_text: str,
) -> str:
    translated = normalize_feature_text(translated_english_text)
    original = normalize_feature_text(original_text)

    if not original:
        return translated
    if not translated:
        return original
    if original.casefold() == translated.casefold():
        return translated  # same text — no duplication needed

    return f"{translated} {original}"
    # e.g.: "road has pothole near junction ರಸ್ತೆಯಲ್ಲಿ ಗುಂಡಿ ಇದೆ"
```

**Why this strategy?**
- English translation is placed **first** to preserve the dominant lexical signal for the TF-IDF model
- The original Kannada/Hindi text is appended to preserve **native civic-domain subword signals**
- The `char_wb` TF-IDF vectoriser extracts character n-grams from both scripts simultaneously
- When original == translation (English input), duplication is avoided

### Text Normalisation

```python
def normalize_feature_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())
```

Whitespace is collapsed while **preserving all Unicode characters** — Kannada (ಕನ್ನಡ), Devanagari (हिंदी), and Latin scripts are all retained without any script-specific filtering.

---

## 9. Civic Translation Glossary

Whisper and general-purpose translation models sometimes mistranslate domain-specific civic terms. A **post-translation correction glossary** patches known errors:

```python
def apply_civic_translation_glossary(text: str, source_language: str) -> str:
    """Patch known civic-term mistranslations for Kannada → English outputs."""
```

**Example corrections (Kannada → English):**
- `"road has pit"` → `"road has pothole"` (ಗುಂಡಿ = pothole, not pit)
- `"drain is full"` → `"drain is blocked"` (ಕಟ್ಟಿಕೊಂಡಿದೆ = blocked/clogged)
- `"light is not"` → `"street light is not working"`

The glossary is applied **after** IndicTrans2 decoding and **before** TF-IDF vectorisation, ensuring the classifier always receives correctly-phrased English text.

---

## 10. Model Training Details

### Dataset
- **Source**: BBMP Grievances 2023 CSV (`data/BBMP_Grievances_2023.csv`) — **207,000+ real complaint records**
- **Cleaned version**: `data/BBMP_cleaned.csv`
- **Text column**: `Sub Category` (short description of the complaint)
- **Label column**: `Category` (BBMP department/type)

### Training Script: `scripts/train_bbmp_model.py`

#### Data Augmentation
The training data is enriched with **supervised augmentation rows** to improve robustness on:
1. **English-language civic terms** not in the original dataset (e.g., `"water fills the road when it rains"` → `Drainage / SWD`)
2. **Transliterated Kannada** input from non-Kannada keyboards
3. **Non-civic inputs** (e.g., `"i played a game yesterday"` → `Non-Civic`) to give the model a rejection class

#### Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    df["classification_text"], df["target"],
    test_size=0.20,
    random_state=42,
    stratify=df["target"],   # maintain class proportions
)
```

#### Class Imbalance Handling
```python
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
clf.fit(X_train_vec, y_train, sample_weight=sample_weights)
```

Rare categories (e.g., `Veterinary`, `Advertisement`) are up-weighted so the classifier does not simply predict majority classes.

### Saved Model Artifact
The serialised model package (`model_bbmp.pkl`, ~1.4 MB) contains:
```python
{
    "vectorizer": FeatureUnion,       # fitted TF-IDF FeatureUnion
    "classifier": MultinomialNB,      # trained classifier
    "category_map": dict,             # raw → normalised category mapping
    "clean_categories": list,         # list of canonical categories
    "trained_at_utc": str,            # ISO timestamp
    "feature_strategy": "english_translation_plus_original_text_word_and_char_tfidf"
}
```

Saved to both `Models/model_bbmp.pkl` (primary) and `model_bbmp.pkl` (legacy fallback). At startup, `load_classifier_assets()` checks both paths.

### Reported Performance
```
Test Accuracy : 98.99%  (on 20% held-out stratified split)
Train set     : ~165,600 complaints
Test set      : ~41,400 complaints
```

---

## 11. NLP Metrics & Energy Monitoring

Every complaint submission writes a row to the `nlp_metrics` table — all values are **measured at runtime**, never estimated or hardcoded.

### NlpMetric Schema (SQLAlchemy)

```python
class NlpMetric(Base):
    id                    = Column(Integer, PrimaryKey)
    complaint_id          = Column(Integer, ForeignKey)
    total_processing_time = Column(Float)   # seconds, wall-clock
    transcription_time    = Column(Float)   # Whisper STT duration
    translation_time      = Column(Float)   # IndicTrans2 / NLLB duration
    classification_time   = Column(Float)   # TF-IDF + NB duration
    ner_time              = Column(Float)   # spaCy NER duration
    zero_shot_time        = Column(Float)   # MNLI fallback duration
    classifier_confidence = Column(Float)   # sklearn predict_proba() score
    zero_shot_triggered   = Column(Boolean) # whether fallback was used
    zero_shot_confidence  = Column(Float)   # MNLI entailment score
    entity_count          = Column(Integer) # len(spacy_doc.ents)
    audio_duration_seconds= Column(Float)   # pydub measurement
    is_duplicate          = Column(Boolean) # duplicate detection flag
    error_stage           = Column(String)  # stage name if exception raised
    total_energy_joules   = Column(Float)   # CPU TDP × processing time
```

### Energy Estimation Methodology

```
Energy (J) = Estimated CPU TDP (W) × Measured Processing Time (s)
```

CPU TDP is derived from `platform.processor()`, `platform.machine()`, and `os.cpu_count()` at startup, classified into power tiers:

| CPU Class | Estimated TDP |
|-----------|--------------|
| ARM (Raspberry Pi, Apple Silicon) | 10 W |
| Mobile / laptop (i3/i5/Ryzen 5) | 15 W |
| High-performance laptop (i7/i9/Ryzen 7) | 45 W |
| Desktop (Core i7/i9 desktop) | 65 W |
| Server / workstation | 95 W |

> **Note**: This is a model-based estimation from CPU TDP specifications, not a direct hardware power measurement (which would require PMU/RAPL access).

---

## 12. Backend Architecture

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Web framework | FastAPI 0.135 + Uvicorn 0.42 |
| ORM | SQLAlchemy 2.0 |
| Data validation | Pydantic 2.12 |
| Authentication | `python-jose` JWT (HS256) + `passlib` bcrypt |
| Audio processing | `pydub` + FFmpeg |
| Image EXIF parsing | `Pillow` |
| Async runtime | Python `asyncio` |

### Key Design Patterns

- **Async endpoints** — all heavy NLP operations run in `asyncio.to_thread()` to keep the event loop non-blocking
- **Singleton model loading** — Whisper, IndicTrans2, spaCy, and the TF-IDF classifier are all loaded **once at startup**, stored in `app.state` or module-level globals
- **Lazy zero-shot loading** — the MNLI model is initialised only on first low-confidence prediction
- **Translation lock** — a single `asyncio.Lock` serialises all translation calls to prevent out-of-memory errors during concurrent requests
- **Structured logging** — all NLP stage timings, confidence scores, and errors are logged via Python `logging` to `bbmp_complaints.log`

### Duplicate Detection (Geo-Semantic)

Duplicate complaints are identified using **both spatial proximity and semantic category matching**:

```
Criteria: same predicted_category
        + haversine_distance(new_lat/lon, existing_lat/lon) < 500 metres
        + submission within 180 days of existing complaint
```

**Haversine formula** (implemented in `haversine_distance_meters()`):
$$d = 2R \arcsin\!\left(\sqrt{\sin^2\!\frac{\Delta\phi}{2} + \cos\phi_1 \cos\phi_2 \sin^2\!\frac{\Delta\lambda}{2}}\right)$$

Where $R = 6{,}371{,}000\,\text{m}$ (mean Earth radius).

---

## 13. Frontend Architecture

| Component | Technology |
|-----------|-----------|
| Framework | React 19 + Vite 8 |
| Routing | React Router v7 |
| Maps | Leaflet + React-Leaflet |
| Charts | Chart.js (bar, line, radar, doughnut, scatter) |
| HTTP client | Fetch API (`services/api.js`) |
| CSS | Vanilla CSS with custom design tokens |

### NLP Analytics Dashboard (`AnalyticsDashboard.jsx`)

The dashboard renders **15+ live charts** from the `/analytics/dashboard` API endpoint:

- **Energy by Stage** — bar chart: transcription vs. translation vs. classification vs. NER energy costs
- **Stage Bottleneck Radar** — radar chart: relative time per stage (highlights translation as the slowest)
- **Classifier Confidence Histogram** — colour-coded: red < 0.50, yellow 0.50–0.85, green > 0.85
- **Category × Language Heatmap** — CSS matrix: cross-tab of complaint category vs. source language
- **NER Entity Count Distribution** — histogram of entities per complaint
- **Audio Duration vs. Processing Time** — scatter plot validating O(n) Whisper scaling
- **Zero-shot Fallback Rate** — doughnut: proportion of complaints requiring MNLI fallback
- **Error Rate by Stage** — identifies which NLP stage is most failure-prone

---

## 14. Database Schema

### Core Tables

```
complaints
├── id, created_at
├── original_text             ← raw Whisper transcript (source language)
├── translated_text           ← English output from IndicTrans2/NLLB
├── category                  ← final routed category (preserved NLP value)
├── predicted_language        ← Whisper detected language
├── confidence                ← classifier confidence score
├── latitude, longitude       ← live GPS from citizen browser
├── trust_level               ← "auto_verified" | "manual_review" (cascades on mismatch)
├── votes                     ← duplicate vote aggregation count
├── status                    ← "pending" | "Verified" | "Rejected" | ...
├── detected_objects          ← JSON string: [{class, confidence, bbox, mask_polygon, mask_area_ratio, severity}]
├── annotated_image_path      ← optional disk path to image with YOLO overlays
├── pothole_severity          ← overall severity: Clear/Low/Medium/High/Severe (nullable)
├── image_suggested_category  ← category suggestion mapped from top detection
└── category_mismatch         ← boolean flag: True if top image detection disagrees with category

nlp_metrics
├── complaint_id              ← FK → complaints.id
├── transcription_time        ← Whisper wall-clock seconds
├── translation_time          ← IndicTrans2 / NLLB seconds
├── classification_time       ← TF-IDF + NB seconds
├── ner_time                  ← spaCy seconds
├── zero_shot_time            ← NLI seconds (0 if not triggered)
├── image_analysis_time       ← YOLO execution latency in seconds
├── detected_object_count     ← count of items found in the image
├── image_model_confidence    ← highest object confidence score
├── classifier_confidence
├── zero_shot_triggered, zero_shot_confidence
├── entity_count
├── audio_duration_seconds
├── total_energy_joules
├── is_duplicate
└── error_stage

complaint_votes
├── complaint_id              ← FK → complaints.id (the canonical complaint)
└── voter_fingerprint         ← hash for deduplication of repeat voters
```├── total_energy_joules
├── is_duplicate
└── error_stageER energy costs
- **Stage Bottleneck Radar** — radar chart: relative time per stage (highlights translation as the slowest)
- **Classifier Confidence Histogram** — colour-coded: red < 0.50, yellow 0.50–0.85, green > 0.85
- **Category × Language Heatmap** — CSS matrix: cross-tab of complaint category vs. source language
- **NER Entity Count Distribution** — histogram of entities per complaint
- **Audio Duration vs. Processing Time** — scatter plot validating O(n) Whisper scaling
- **Zero-shot Fallback Rate** — doughnut: proportion of complaints requiring MNLI fallback
- **Error Rate by Stage** — identifies which NLP stage is most failure-prone

---

## 14. Database Schema

### Core Tables

```
complaints
├── id, created_at
├── original_text       ← raw Whisper transcript (source language)
├── translated_text     ← English output from IndicTrans2/NLLB
├── predicted_category  ← TF-IDF + NB output
├── predicted_language  ← Whisper detected language
├── confidence          ← classifier predict_proba score
├── latitude, longitude ← live GPS from citizen browser
├── trust_level         ← "auto_verified" | "manual_review"
├── votes               ← duplicate vote aggregation count
└── is_verified         ← HITL admin approval flag

nlp_metrics
├── complaint_id        ← FK → complaints.id
├── transcription_time  ← Whisper wall-clock seconds
├── translation_time    ← IndicTrans2 / NLLB seconds
├── classification_time ← TF-IDF + NB seconds
├── ner_time            ← spaCy seconds
├── zero_shot_time      ← MNLI seconds (0 if not triggered)
├── classifier_confidence
├── zero_shot_triggered, zero_shot_confidence
├── entity_count
├── audio_duration_seconds
├── total_energy_joules
├── is_duplicate
└── error_stage

complaint_votes
├── complaint_id        ← FK → complaints.id (the canonical complaint)
└── voter_fingerprint   ← hash for deduplication of repeat voters
```

**Database**: PostgreSQL 14+ (primary) · SQLite (automatic fallback)

---

## 15. Deployment & DevOps

### Docker Compose

```yaml
services:
  db:   # PostgreSQL 14
  app:  # FastAPI + Uvicorn (port 8000)
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | JWT signing secret (min. 32 chars) |
| `JWT_ALGORITHM` | `HS256` |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | Admin portal credentials |
| `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME` | PostgreSQL connection |
| `ENABLE_ZERO_SHOT_FALLBACK` | `"true"` / `"false"` (default `true`) |
| `ZERO_SHOT_MODEL_NAME` | Override MNLI model (default `valhalla/distilbart-mnli-12-1`) |
| `ZERO_SHOT_MIN_CONFIDENCE` | Primary classifier confidence threshold (default `0.85`) |
| `ZERO_SHOT_MIN_SCORE` | MNLI minimum score for override (default `0.55`) |
| `ZERO_SHOT_SPARSE_MIN_SCORE` | MNLI score for sparse inputs (default `0.60`) |
| `YOLO_MODEL_PATH` | Path override for image segmentation weights |
| `IMAGE_RECONCILE_CONFIDENCE_THRESHOLD` | Disagreement threshold for classification reconciliation (default `0.6`) |

### Cloud Deployment
- **Railway**: configured via `railway.toml`
- Backend port: `8000` (Uvicorn)
- Frontend port: `5173` (Vite dev) / static build for production

---

## Appendix — NLP Library Dependency Tree

```
openai-whisper 20250625
  └── torch 2.10.0+cpu
  └── torchaudio 2.10.0+cpu
  └── tiktoken 0.12.0
  └── numba 0.64.0

transformers 4.57.6
  └── torch 2.10.0+cpu
  └── tokenizers 0.22.2 (Rust-based HuggingFace fast tokenizers)
  └── sentencepiece 0.2.1 (SentencePiece subword model)
  └── safetensors 0.7.0 (model weight loading)
  └── huggingface_hub 0.36.2

scikit-learn 1.7.2
  └── scipy 1.17.1 (sparse matrix operations for TF-IDF)
  └── numpy 2.4.3
  └── joblib 1.5.3 (parallel feature extraction)

spacy 3.8.11
  └── en_core_web_sm 3.8.0 (tok2vec + NER pipeline)
  └── thinc 8.3.11 (spaCy's ML backend)
  └── blis 1.3.3 (BLAS-level matrix ops)
  └── murmurhash 1.0.15 (feature hashing)
```

---

## 16. Stage 6 · Image Analysis & Reconciliation (YOLOv8n-seg)

This stage adds **cross-modal verification** by checking visual evidence against the text-based department routing.

### Model Specs & Config
- **Model Architecture**: Ultralytics YOLOv8n-seg (segmentation)
- **Loaded at startup** as a module-level singleton from `Models/civic_multiclass_seg_best.pt` (overridable via `YOLO_MODEL_PATH` env var).
- **Execution**: Offloaded to a separate CPU thread via `asyncio.to_thread` to ensure zero blocking on the main FastAPI event loop.
- **Image Resizing**: Input images are dynamically resized to a maximum dimension of `640px` (maintaining aspect ratio) before prediction to optimize CPU inference latency.

### Visually Detectable Classes (8 Mappings)
The visual model outputs segmentations and classes, which are mapped to canon BBMP department categories via the `DETECTION_CLASS_TO_CATEGORY` mapping matrix:

| YOLO Class | Maps to Department | Severity Rule |
|------------|---------------------|---------------|
| `pothole` | `Road Repair` | Mask area vs Image area ratio |
| `garbage_pile` | `Garbage / Sanitation` | Mask area vs Image area ratio |
| `broken_streetlight` | `Street Light` | Presence-based |
| `waterlogging` | `Drainage / SWD` | Mask area vs Image area ratio |
| `damaged_drain` | `Drainage / SWD` | Presence-based |
| `illegal_hoarding` | `Advertisement` | Presence-based |
| `overgrown_park` | `Parks / Forest` | Presence-based |
| `water_leak` | `Water Supply` | Presence-based |

*Other categories (e.g. Revenue, Non-Civic, Veterinary, Town Planning, Traffic, Health/Sanitation, Others) are administrative/textual in nature and explicitly excluded from image analysis mapping.*

### Coordinate Normalization
All bounding box (`bbox` coordinate arrays `[xmin, ymin, xmax, ymax]`) and segmentation boundaries (`mask_polygon` coordinates `[[x1, y1], [x2, y2], ...]`) are normalized to a `0.0 - 1.0` range relative to the inference image canvas dimensions, ensuring they align perfectly with any UI screen size in the admin overlay.

### Cross-Modal Reconciliation Logic
The pipeline protects against silent category hijacking by using the image analysis strictly for **validation/mismatch checking** and **trust level overrides**, never overwriting the text classifier’s `category`.

```
                  ┌───────────────────────────────┐
                  │ EXIF Metadata Verification   │
                  │ (Live GPS vs Image Geo-Tags)  │
                  └──────────────┬────────────────┘
                                 │ Pass
                                 ▼
                     Status = "Verified"
                     Trust Level = "high"
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │ Stage 6: Image Analysis       │
                  │ (YOLOv8n-seg Prediction)      │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
         Disagreement Check:
         - Top visual detection confidence > threshold (default 0.6)
         - Image-suggested category != text-predicted category
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼ Yes                           ▼ No
        ┌─────────────────┐             ┌───────────────────┐
        │ Mismatch Flag   │             │ Retain high trust │
        │ Status="pending"│             │ Keep Verified status│
        │ Trust="manual_   │             └───────────────────┘
        │ review"         │
        └─────────────────┘
```

When a mismatch is flagged:
1. `category_mismatch` is set to `True`.
2. `image_suggested_category` is populated with the mapped category.
3. The complaint is downgraded to `status = "pending"` and `trust_level = "manual_review"`, ensuring human eyes audit the issue in the admin console.

---

## 17. Known Issues & Ongoing Investigations

- **IndicTransToolkit Compilation on Windows**: The `IndicTrans2` model requires `IndicTransToolkit` to handle accurate script-level tokenization for Indian languages. On Windows, installing this toolkit requires Microsoft C++ Build Tools (`processor.pyx`, `processor.c`). Without it, the application falls back to a generic processor which currently induces extreme hallucination failures for Kannada translation (e.g., continuous repetition of English phrases).
- **English "Garbage" Anomaly**: A specific test scenario (*"Garbage collection has not happened for five days in our area. Waste is piled up..."*) is currently being misclassified as `Others` (confidence ~88%) instead of `Garbage / Sanitation`. This is logged for investigation to determine if it requires a `ZERO_SHOT_MIN_CONFIDENCE` tweak or dataset reinforcement.
