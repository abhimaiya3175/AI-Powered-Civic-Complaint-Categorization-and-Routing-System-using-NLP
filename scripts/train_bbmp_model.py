import pandas as pd
import re
import pickle
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

TEXT_COLUMN = "Sub Category"
CATEGORY_COLUMN = "Category"

category_map = {
    "Electrical": "Street Light",
    "Solid Waste (Garbage) Related": "Garbage / Sanitation",
    "Road Maintenance(Engg)": "Road Repair",
    "Road Infrastructure": "Road Repair",
    "Storm Water Drain(SWD)": "Drainage / SWD",
    "Sanitation": "Garbage / Sanitation",
    "Health Dept": "Health / Sanitation",
    "Water Crisis": "Water Supply",
    "Parks and Play grounds": "Parks",
    "Forest": "Parks / Forest",
    "Town Planning": "Town Planning",
    "Revenue Department": "Revenue",
    "veterinary": "Veterinary",
    "Advertisement": "Advertisement",
    "Others": "Others"
}

# ====================== LOAD ======================
print("Loading BBMP data...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TEXT_COLUMN, CATEGORY_COLUMN])

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
df["target"] = df[CATEGORY_COLUMN].map(category_map).fillna("Others")

print(f"Dataset size: {len(df):,} complaints")
print("Class distribution:\n", df["target"].value_counts())

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
    "clean_categories": list(df["target"].unique())
}

with open("model_bbmp.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("\n✅ SUCCESS! Model saved as model_bbmp.pkl")
print("Ready for FastAPI + Voice pipeline!")