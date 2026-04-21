import pandas as pd
import re
import pickle
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text)

# ====================== CONFIG (SAFE) ======================
CSV_PATH = "data/BBMP_cleaned.csv"        # ← Change only if your file name is different
MODEL_OUTPUT_PATH = Path("Models/model_bbmp.pkl")
LEGACY_OUTPUT_PATH = Path("model_bbmp.pkl")

TEXT_COLUMN = "Sub Category"
CATEGORY_COLUMN = "Category"


def normalize_category(value):
    raw = str(value).strip().lower()
    return re.sub(r'\s+', ' ', raw)


category_map = {
    "electrical": "Street Light",
    "solid waste (garbage) related": "Garbage / Sanitation",
    "road maintenance(engg)": "Road Repair",
    "road infrastructure": "Road Repair",
    "storm water drain(swd)": "Drainage / SWD",
    "sanitation": "Garbage / Sanitation",
    "health dept": "Health / Sanitation",
    "water crisis": "Water Supply",
    "parks and play grounds": "Parks",
    "forest": "Parks / Forest",
    "town planning": "Town Planning",
    "revenue department": "Revenue",
    "veterinary": "Veterinary",
    "advertisement": "Advertisement",
    "traffic engineer cell (tec)": "Traffic",
    "others": "Others"
}

# ====================== LOAD ======================
print("Loading BBMP data...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TEXT_COLUMN, CATEGORY_COLUMN])

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
df["category_normalized"] = df[CATEGORY_COLUMN].apply(normalize_category)
df["target"] = df["category_normalized"].map(category_map).fillna("Others")

print(f"Dataset size: {len(df):,} complaints")
print("Class distribution:\n", df["target"].value_counts())

unmapped_categories = (
    df.loc[~df["category_normalized"].isin(category_map.keys()), CATEGORY_COLUMN]
    .value_counts()
    .head(10)
)
if not unmapped_categories.empty:
    print("\nTop unmapped raw categories (fell back to 'Others'):\n", unmapped_categories)

# Augment training data to correct specific predictions
augmentation_data = pd.DataFrame([
    {"Sub Category": "water fills the road when it rains", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    {"Sub Category": "water logging filling on road due to heavy rain", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    {"Sub Category": "traffic jam is very high on this route", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    {"Sub Category": "heavy traffic jam vehicles slow moving", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    # Add transliterated kannada forms and english words as a fallback
    {"Sub Category": "ಮಳೆ ಬಂದಾಗ ರಸ್ತೆಯಲ್ಲಿ ನೀರು ತುಂಬಿಕೊಳ್ಳುತ್ತದೆ", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    {"Sub Category": "ಈ ಮಾರ್ಗದಲ್ಲಿ ಟ್ರಾಫಿಕ್ ಜಾಮ್ ತುಂಬಾ ಜಾಸ್ತಿ ಇದೆ", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    # Non-Civic irrelevant inputs
    {"Sub Category": "i played a game yesterday with my friends", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "hello how are you doing today", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "what is the weather like in bangalore", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "i want to order food from swiggy", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "this is a test audio recording testing 1 2 3", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "i went to the mall to buy clothes", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "my favorite movie is playing in theatres", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "good morning have a nice day", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "who won the cricket match yesterday rcb", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "just checking if the microphone works", "Category": "Non-Civic", "target": "Non-Civic"},
]*50) # duplicate them a bunch of times to give enough weight

augmentation_data[TEXT_COLUMN] = augmentation_data["Sub Category"].apply(clean_text)
df = pd.concat([df, augmentation_data], ignore_index=True)

# ====================== SPLIT + TRAIN ======================
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN], df["target"], test_size=0.20, random_state=42, stratify=df["target"]
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=2, max_df=0.95)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train, sample_weight=sample_weights)

# ====================== RESULTS ======================
y_pred = clf.predict(X_test_vec)

print("\n" + "="*60)
print("MODEL PERFORMANCE (BBMP 2024 Data)")
print("="*60)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}  ← This should be ~0.96+")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ====================== SAVE ======================
model_package = {
    "vectorizer": vectorizer,
    "classifier": clf,
    "category_map": category_map,
    "clean_categories": list(df["target"].unique()),
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "text_column": TEXT_COLUMN,
    "category_column": CATEGORY_COLUMN,
}

MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with MODEL_OUTPUT_PATH.open("wb") as f:
    pickle.dump(model_package, f)
with LEGACY_OUTPUT_PATH.open("wb") as f:
    pickle.dump(model_package, f)

print(f"\n✅ SUCCESS! Model saved to {MODEL_OUTPUT_PATH} (runtime) and {LEGACY_OUTPUT_PATH} (legacy)")
print("Ready for FastAPI + Voice pipeline!")