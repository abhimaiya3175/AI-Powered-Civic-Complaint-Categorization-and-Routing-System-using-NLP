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
    {"Sub Category": "what time is it in london right now", "Category": "Non-Civic", "target": "Non-Civic"},
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

# ── Augmentation for underrepresented categories ─────────────────────
# Traffic (296 real samples), Water Supply (700), Drainage (1677),
# Revenue (1062), Advertisement (854), Parks (1130), Town Planning (1813)
underrepresented_augmentation_rows = [
    # ── Traffic (most underrepresented: 296 real samples) ────────────
    {"Sub Category": "traffic signal not working at junction", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    {"Sub Category": "traffic jam every day during peak hours", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    {"Sub Category": "no traffic signal at busy intersection", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    {"Sub Category": "traffic police needed at school crossing", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    {"Sub Category": "broken traffic signal causing accidents", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    {"Sub Category": "vehicles parked illegally blocking traffic flow", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    {"Sub Category": "speed breaker missing on busy road causing speeding", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    {"Sub Category": "traffic congestion due to road narrowing", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    {"Sub Category": "ट्रैफिक सिग्नल खराब है चौराहे पर", "Translated English": "traffic signal broken at intersection", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    {"Sub Category": "ಟ್ರಾಫಿಕ್ ಸಿಗ್ನಲ್ ಕೆಟ್ಟಿದೆ ಜಂಕ್ಷನ್ ಬಳಿ", "Translated English": "traffic signal broken near junction", "Category": "Traffic Engineer Cell (TEC)", "target": "Traffic"},
    # ── Water Supply (700 real samples) ──────────────────────────────
    {"Sub Category": "no water supply in our area for two days", "Category": "Water Crisis", "target": "Water Supply"},
    {"Sub Category": "water not coming from tap pipeline issue", "Category": "Water Crisis", "target": "Water Supply"},
    {"Sub Category": "water pipeline burst leaking water everywhere", "Category": "Water Crisis", "target": "Water Supply"},
    {"Sub Category": "contaminated water coming from corporation tap", "Category": "Water Crisis", "target": "Water Supply"},
    {"Sub Category": "low water pressure in our locality", "Category": "Water Crisis", "target": "Water Supply"},
    {"Sub Category": "water supply timing is irregular and insufficient", "Category": "Water Crisis", "target": "Water Supply"},
    {"Sub Category": "पानी सप्लाई बंद है दो दिन से", "Translated English": "water supply stopped for two days", "Category": "Water Crisis", "target": "Water Supply"},
    {"Sub Category": "ನೀರು ಸರಬರಾಜು ಇಲ್ಲ ಎರಡು ದಿನ", "Translated English": "no water supply for two days", "Category": "Water Crisis", "target": "Water Supply"},
    # ── Drainage / SWD (1677 real samples) ───────────────────────────
    {"Sub Category": "drainage blocked and sewage overflowing onto road", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    {"Sub Category": "storm water drain clogged causing flooding", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    {"Sub Category": "road flooded due to blocked drainage", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    {"Sub Category": "sewage water stagnating near houses bad smell", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    {"Sub Category": "waterlogging on main road after rain", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    {"Sub Category": "open manhole cover missing near drain", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    {"Sub Category": "नाली बंद है गंदा पानी सड़क पर", "Translated English": "drain blocked dirty water on road", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    {"Sub Category": "ಚರಂಡಿ ಬ್ಲಾಕ್ ಆಗಿ ನೀರು ರಸ್ತೆಗೆ", "Translated English": "drain blocked water on road", "Category": "Storm  Water Drain(SWD)", "target": "Drainage / SWD"},
    # ── Revenue (1062 real samples) ──────────────────────────────────
    {"Sub Category": "property tax khata transfer not done", "Category": "Revenue Department", "target": "Revenue"},
    {"Sub Category": "khata certificate application pending since months", "Category": "Revenue Department", "target": "Revenue"},
    {"Sub Category": "property tax bill incorrect need correction", "Category": "Revenue Department", "target": "Revenue"},
    {"Sub Category": "mutation of property records not updated", "Category": "Revenue Department", "target": "Revenue"},
    {"Sub Category": "ಖಾತಾ ವರ್ಗಾವಣೆ ಆಗಿಲ್ಲ", "Translated English": "khata transfer not done", "Category": "Revenue Department", "target": "Revenue"},
    {"Sub Category": "खाता ट्रांसफर नहीं हुआ", "Translated English": "khata transfer not done", "Category": "Revenue Department", "target": "Revenue"},
    # ── Advertisement (854 real samples) ─────────────────────────────
    {"Sub Category": "illegal advertisement hoarding on footpath", "Category": "Advertisement", "target": "Advertisement"},
    {"Sub Category": "unauthorized flex banners blocking road view", "Category": "Advertisement", "target": "Advertisement"},
    {"Sub Category": "illegal posters pasted on public walls", "Category": "Advertisement", "target": "Advertisement"},
    {"Sub Category": "commercial billboard without permission near school", "Category": "Advertisement", "target": "Advertisement"},
    {"Sub Category": "ಅಕ್ರಮ ಜಾಹೀರಾತು ಹೋರ್ಡಿಂಗ್", "Translated English": "illegal advertisement hoarding", "Category": "Advertisement", "target": "Advertisement"},
    {"Sub Category": "अवैध विज्ञापन होर्डिंग", "Translated English": "illegal advertisement hoarding", "Category": "Advertisement", "target": "Advertisement"},
    # ── Parks (1130 real samples) ────────────────────────────────────
    {"Sub Category": "park playground swings are broken dangerous for children", "Category": "Parks and Play grounds", "target": "Parks"},
    {"Sub Category": "public park not maintained grass overgrown", "Category": "Parks and Play grounds", "target": "Parks"},
    {"Sub Category": "park benches broken and lights not working", "Category": "Parks and Play grounds", "target": "Parks"},
    {"Sub Category": "playground equipment rusted needs replacement", "Category": "Parks and Play grounds", "target": "Parks"},
    {"Sub Category": "ಉದ್ಯಾನವನ ಹಾಳಾಗಿದೆ ಮಕ್ಕಳ ಆಟ ಸಾಮಗ್ರಿ ಮುರಿದಿವೆ", "Translated English": "park damaged children playground equipment broken", "Category": "Parks and Play grounds", "target": "Parks"},
    {"Sub Category": "पार्क खराब हालत में है झूले टूटे", "Translated English": "park in bad condition swings broken", "Category": "Parks and Play grounds", "target": "Parks"},
    # ── Town Planning (1813 real samples) ────────────────────────────
    {"Sub Category": "unauthorized building construction in residential area", "Category": "Town Planning", "target": "Town Planning"},
    {"Sub Category": "illegal commercial construction without building plan approval", "Category": "Town Planning", "target": "Town Planning"},
    {"Sub Category": "encroachment on public footpath by shop owner", "Category": "Town Planning", "target": "Town Planning"},
    {"Sub Category": "building violating setback norms in our layout", "Category": "Town Planning", "target": "Town Planning"},
    {"Sub Category": "ಅನಧಿಕೃತ ಕಟ್ಟಡ ನಿರ್ಮಾಣ", "Translated English": "unauthorized building construction", "Category": "Town Planning", "target": "Town Planning"},
    {"Sub Category": "अवैध निर्माण बिना अनुमति", "Translated English": "illegal construction without permission", "Category": "Town Planning", "target": "Town Planning"},
    # ── Non-Civic (0 real samples, only synthetic) ───────────────────
    {"Sub Category": "what is the score of the cricket match today", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "i want to book a movie ticket for tonight", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "where is the nearest good restaurant", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "how is the weather looking for tomorrow", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "my phone internet is very slow", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "आज क्रिकेट का क्या स्कोर है", "Translated English": "what is the cricket score today", "Category": "Non-Civic", "target": "Non-Civic"},
    {"Sub Category": "ನನಗೆ ಸಿನಿಮಾ ಟಿಕೆಟ್ ಬೇಕು", "Translated English": "i want a movie ticket", "Category": "Non-Civic", "target": "Non-Civic"},
]

augmentation_data = pd.DataFrame(
    (english_augmentation_rows * 50)
    + (multilingual_augmentation_rows * 30)
    + (road_native_augmentation_rows * 80)
    + (underrepresented_augmentation_rows * 100)  # Heavy augmentation for underrepresented categories
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
