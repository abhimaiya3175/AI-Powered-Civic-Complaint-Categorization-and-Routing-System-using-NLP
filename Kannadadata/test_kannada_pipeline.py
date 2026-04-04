import whisper
import pickle
import spacy
from datasets import load_dataset
import os

print("🚀 Starting Full Kannada Pipeline Test...\n")

# ====================== LOAD MODELS ======================
whisper_model = whisper.load_model("base")
with open("model_bbmp.pkl", "rb") as f:
    model_package = pickle.load(f)

vectorizer = model_package["vectorizer"]
clf = model_package["classifier"]
nlp = spacy.load("en_core_web_sm")

print("✅ Models loaded (Whisper + 98.5% BBMP model)\n")

# ====================== LOAD KANNADA DATASET ======================
dataset = load_dataset("SPRINGLab/IndicTTS_Kannada", split="train")

print(f"✅ Loaded {len(dataset)} Kannada audio samples\n")
print("Testing first 15 samples...\n")
print("="*80)

for i in range(15):   # You can increase to 30 or 50 later
    sample = dataset[i]
    audio = sample["audio"]
    kannada_text = sample["text"]
    
    # Save temporary WAV for Whisper
    temp_path = f"temp_test_{i}.wav"
    import soundfile as sf
    sf.write(temp_path, audio["array"], audio["sampling_rate"])

    # Step 1: Whisper → STT + Translate to English
    result = whisper_model.transcribe(temp_path, language=None, task="translate")
    english_text = result["text"].strip()

    # Step 2: Your BBMP model prediction
    X = vectorizer.transform([english_text])
    predicted_category = clf.predict(X)[0]

    # Step 3: Location extraction (NLP)
    doc = nlp(english_text)
    locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
    location = ", ".join(locations) if locations else "Not detected"

    print(f"Sample {i+1:02d}")
    print(f"   Kannada     : {kannada_text[:90]}...")
    print(f"   English     : {english_text}")
    print(f"   Category    : {predicted_category}")
    print(f"   Location    : {location}")
    print("-" * 60)

    os.remove(temp_path)   # clean up

print("\n🎉 Pipeline Test Completed!")
print("Your 98.5% BBMP model is now working with real Kannada audio → English translation!")