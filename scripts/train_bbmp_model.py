import pandas as pd
import re
import pickle
import sys
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp_features import build_multilingual_classification_text

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
TRANSLATED_TEXT_COLUMN = "Translated English"


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
df["translated_english_text"] = df[TEXT_COLUMN]
df["classification_text"] = [
    build_multilingual_classification_text(original, translated)
    for original, translated in zip(df[TEXT_COLUMN], df["translated_english_text"])
]

print(f"Dataset size: {len(df):,} complaints")
print("Class distribution:\n", df["target"].value_counts())

unmapped_categories = (
    df.loc[~df["category_normalized"].isin(category_map.keys()), CATEGORY_COLUMN]
    .value_counts()
    .head(10)
)
if not unmapped_categories.empty:
    print("\nTop unmapped raw categories (fell back to 'Others'):\n", unmapped_categories)

# Augment training data with supervised examples, not inference-time rules.
english_augmentation_rows = [
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
]

multilingual_augmentation_rows = [
    {
        "Sub Category": "हमारे क्षेत्र में सड़क पर गड्ढे होने के कारण लोगों को परेशानी हो रही है",
        "Translated English": "there are potholes on the road in our area causing trouble to people",
        "Category": "Road Maintenance(Engg)",
        "target": "Road Repair",
    },
    {
        "Sub Category": "मुख्य सड़क पर बड़े गड्ढे हैं और वाहन चलाना मुश्किल है",
        "Translated English": "there are large potholes on the main road and driving is difficult",
        "Category": "Road Maintenance(Engg)",
        "target": "Road Repair",
    },
    {
        "Sub Category": "सड़क टूटी हुई है और मरम्मत की जरूरत है",
        "Translated English": "the road is broken and needs repair",
        "Category": "Road Maintenance(Engg)",
        "target": "Road Repair",
    },
    {
        "Sub Category": "ರಸ್ತೆಯಲ್ಲಿ ಗುಂಡಿಗಳು ಇರುವುದರಿಂದ ಜನರಿಗೆ ತೊಂದರೆ ಆಗುತ್ತಿದೆ",
        "Translated English": "there are potholes on the road causing problems for people",
        "Category": "Road Maintenance(Engg)",
        "target": "Road Repair",
    },
    {
        "Sub Category": "हमारी गली की स्ट्रीट लाइट बंद है",
        "Translated English": "the street light in our lane is not working",
        "Category": "Electrical",
        "target": "Street Light",
    },
    {
        "Sub Category": "कचरा तीन दिन से नहीं उठाया गया है",
        "Translated English": "garbage has not been collected for three days",
        "Category": "Solid Waste (Garbage) Related",
        "target": "Garbage / Sanitation",
    },
    {
        "Sub Category": "नाली जाम है और गंदा पानी सड़क पर आ रहा है",
        "Translated English": "the drain is blocked and dirty water is overflowing on the road",
        "Category": "Storm  Water Drain(SWD)",
        "target": "Drainage / SWD",
    },
    {
        "Sub Category": "हमारे घर में पानी की सप्लाई नहीं आ रही है",
        "Translated English": "there is no water supply at our house",
        "Category": "Water Crisis",
        "target": "Water Supply",
    },
    {
        "Sub Category": "इस रास्ते पर बहुत ज्यादा ट्रैफिक जाम है",
        "Translated English": "there is heavy traffic jam on this route",
        "Category": "Traffic Engineer Cell (TEC)",
        "target": "Traffic",
    },
    {
        "Sub Category": "पार्क में झूले टूटे हुए हैं",
        "Translated English": "the playground equipment in the park is broken",
        "Category": "Parks and Play Grounds",
        "target": "Parks",
    },
    {
        "Sub Category": "अवैध होर्डिंग सड़क के किनारे लगाया गया है",
        "Translated English": "an illegal advertisement hoarding has been placed beside the road",
        "Category": "Advertisement",
        "target": "Advertisement",
    },
    {
        "Sub Category": "आवारा कुत्ते लोगों को काट रहे हैं",
        "Translated English": "stray dogs are biting people",
        "Category": "Veterinary",
        "target": "Veterinary",
    },
    {
        "Sub Category": "मच्छरों की समस्या बहुत ज्यादा है",
        "Translated English": "there is a severe mosquito problem",
        "Category": "Health Dept",
        "target": "Health / Sanitation",
    },
    {
        "Sub Category": "मैंने कल दोस्तों के साथ खेल खेला",
        "Translated English": "i played a game with friends yesterday",
        "Category": "Non-Civic",
        "target": "Non-Civic",
    },
]

road_native_augmentation_rows = [
    {
        "Sub Category": "हमारे क्षेत्र में सड़क पर गड्ढे होने के कारण लोगों को परेशानी हो रही है",
        "Translated English": "",
        "Category": "Road Maintenance(Engg)",
        "target": "Road Repair",
    },
    {
        "Sub Category": "सड़क पर गड्ढे हैं और लोगों को परेशानी हो रही है",
        "Translated English": "",
        "Category": "Road Maintenance(Engg)",
        "target": "Road Repair",
    },
    {
        "Sub Category": "गली की सड़क में गड्ढे हैं",
        "Translated English": "",
        "Category": "Road Maintenance(Engg)",
        "target": "Road Repair",
    },
    {
        "Sub Category": "मुख्य सड़क पर गड्ढों की मरम्मत चाहिए",
        "Translated English": "",
        "Category": "Road Maintenance(Engg)",
        "target": "Road Repair",
    },
]

augmentation_data = pd.DataFrame(
    (english_augmentation_rows * 50)
    + (multilingual_augmentation_rows * 30)
    + (road_native_augmentation_rows * 80)
)

augmentation_data[TEXT_COLUMN] = augmentation_data["Sub Category"].apply(clean_text)
augmentation_data[TRANSLATED_TEXT_COLUMN] = augmentation_data.get(
    TRANSLATED_TEXT_COLUMN,
    augmentation_data[TEXT_COLUMN],
).fillna(augmentation_data[TEXT_COLUMN])
augmentation_data["translated_english_text"] = augmentation_data[TRANSLATED_TEXT_COLUMN].apply(clean_text)
augmentation_data["category_normalized"] = augmentation_data[CATEGORY_COLUMN].apply(normalize_category)
augmentation_data["classification_text"] = [
    build_multilingual_classification_text(original, translated)
    for original, translated in zip(
        augmentation_data[TEXT_COLUMN],
        augmentation_data["translated_english_text"],
    )
]

native_only_augmentation_data = pd.DataFrame(multilingual_augmentation_rows * 80)
native_only_augmentation_data[TEXT_COLUMN] = native_only_augmentation_data[TEXT_COLUMN].apply(clean_text)
native_only_augmentation_data[TRANSLATED_TEXT_COLUMN] = ""
native_only_augmentation_data["translated_english_text"] = ""
native_only_augmentation_data["category_normalized"] = native_only_augmentation_data[CATEGORY_COLUMN].apply(normalize_category)
native_only_augmentation_data["classification_text"] = native_only_augmentation_data[TEXT_COLUMN]

augmentation_data = pd.concat(
    [augmentation_data, native_only_augmentation_data],
    ignore_index=True,
)
df = pd.concat([df, augmentation_data], ignore_index=True)

# ====================== SPLIT + TRAIN ======================
X_train, X_test, y_train, y_test = train_test_split(
    df["classification_text"], df["target"], test_size=0.20, random_state=42, stratify=df["target"]
)

vectorizer = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        ),
    ),
    (
        "char_wb",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.98,
            max_features=12000,
            sublinear_tf=True,
        ),
    ),
])
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
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}  (target: ~0.96+)")
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
    "classification_text_column": "classification_text",
    "category_column": CATEGORY_COLUMN,
    "feature_strategy": "english_translation_plus_original_text_word_and_char_tfidf",
}

MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with MODEL_OUTPUT_PATH.open("wb") as f:
    pickle.dump(model_package, f)
with LEGACY_OUTPUT_PATH.open("wb") as f:
    pickle.dump(model_package, f)

print(f"\nSUCCESS! Model saved to {MODEL_OUTPUT_PATH} (runtime) and {LEGACY_OUTPUT_PATH} (legacy)")
print("Ready for FastAPI + Voice pipeline!")
