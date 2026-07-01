"""
test_ml_model_and_languages.py
=================================
Comprehensive test to prove:
  1. The trained TF-IDF + Naive Bayes ML model is actually used for predictions
  2. ALL 13 civic categories classify correctly, in ALL 3 languages
     (English, Kannada, Hindi) — plus Non-Civic rejection
  3. Feature explanations are clean and human-readable (no char_wb__ junk)
  4. The full NLP pipeline: Transcribe → Translate → Classify works end-to-end

Categories covered (per BBMP department mapping):
  Road Repair, Street Light, Garbage / Sanitation, Water Supply,
  Drainage / SWD, Health / Sanitation, Parks, Traffic, Town Planning,
  Revenue, Veterinary, Advertisement, Others, Non-Civic (rejected)

Run:
    python tests/test_ml_model_and_languages.py
"""

import requests
import json
import time
import sys
import os
from datetime import datetime, timezone

# Force UTF-8 output on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


BASE_URL = "http://localhost:8000"

# ─── Bangalore GPS base with per-run offset ──────────────────────────────────
_BASE_LAT = 12.9716
_BASE_LON = 77.5946
_run_seed = int(time.time()) % 1000
_RUN_LAT_OFFSET = 0.020 + (_run_seed % 20) * 0.001
_RUN_LON_OFFSET = 0.020 + (_run_seed // 20) * 0.001
TEST_LATITUDE  = _BASE_LAT + _RUN_LAT_OFFSET
TEST_LONGITUDE = _BASE_LON + _RUN_LON_OFFSET


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES — Real complaints in English, Kannada, and Hindi
# Every civic category is tested in every language.
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [

    # ── ROAD REPAIR ───────────────────────────────────────────────────────────
    {
        "name": "EN → Road Repair (pothole)",
        "text": (
            "There are multiple large potholes on the main road near HSR Layout. "
            "The road surface is completely damaged and vehicles are getting stuck. "
            "Road repair work is urgently needed to fix these potholes."
        ),
        "language": "en",
        "expected_category": "Road Repair",
    },
    {
        "name": "KN → Road Repair (ರಸ್ತೆ ಗುಂಡಿ)",
        "text": (
            "ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಗುಂಡಿಗಳಿವೆ. ರಸ್ತೆ ರಿಪೇರಿ ಮಾಡಬೇಕು. "
            "ವಾಹನಗಳು ಓಡಾಡಲು ಕಷ್ಟವಾಗುತ್ತಿದೆ. ರಸ್ತೆ ಹಾಳಾಗಿದೆ ಕೂಡಲೇ ರಿಪೇರಿ ಮಾಡಿ."
        ),
        "language": "kn",
        "expected_category": "Road Repair",
    },
    {
        "name": "HI → Road Repair (सड़क गड्ढा)",
        "text": (
            "सड़क में बड़े गड्ढे हैं। सड़क की मरम्मत करनी चाहिए। "
            "गाड़ियां चलाना मुश्किल हो रहा है। सड़क टूटी हुई है तुरंत मरम्मत करें।"
        ),
        "language": "hi",
        "expected_category": "Road Repair",
    },

    # ── STREET LIGHT ──────────────────────────────────────────────────────────
    {
        "name": "EN → Street Light (not working)",
        "text": (
            "The street light on 80 feet road near Indiranagar has not been working for three weeks. "
            "The entire stretch is pitch dark at night causing accidents. "
            "Street light electrical maintenance is urgently needed."
        ),
        "language": "en",
        "expected_category": "Street Light",
    },
    {
        "name": "KN → Street Light (ಬೀದಿ ದೀಪ)",
        "text": (
            "ನಮ್ಮ ಬೀದಿಯಲ್ಲಿ ಬೀದಿ ದೀಪ ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ. "
            "ರಾತ್ರಿ ಕತ್ತಲೆಯಲ್ಲಿ ನಡೆಯಲು ಭಯವಾಗುತ್ತಿದೆ. "
            "ವಿದ್ಯುತ್ ದೀಪ ದುರಸ್ತಿ ಮಾಡಬೇಕು."
        ),
        "language": "kn",
        "expected_category": "Street Light",
    },
    {
        "name": "HI → Street Light (बिजली बत्ती)",
        "text": (
            "हमारी गली में स्ट्रीट लाइट काम नहीं कर रही है। "
            "रात में अंधेरा रहता है बहुत खतरनाक है। "
            "बिजली की बत्ती ठीक करवाओ।"
        ),
        "language": "hi",
        "expected_category": "Street Light",
    },

    # ── GARBAGE / SANITATION ──────────────────────────────────────────────────
    {
        "name": "EN → Garbage / Sanitation (waste piled up)",
        "text": (
            "Garbage collection has not happened for five days in our area. "
            "Waste is piled up and garbage bins are overflowing. "
            "Garbage truck not coming. Sanitation issue near Koramangala."
        ),
        "language": "en",
        "expected_category": "Garbage / Sanitation",
    },
    {
        "name": "KN → Garbage / Sanitation (ಕಸ ಸಮಸ್ಯೆ)",
        "text": (
            "ನಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ಐದು ದಿನಗಳಿಂದ ಕಸ ಸಂಗ್ರಹಣೆ ಆಗಿಲ್ಲ. "
            "ಕಸದ ಬುಟ್ಟಿಗಳು ತುಂಬಿ ಹೋಗಿವೆ. ಕಸದ ಗಾಡಿ ಬರುತ್ತಿಲ್ಲ. "
            "ಸ್ವಚ್ಛತಾ ಸಮಸ್ಯೆ ಕೋರಮಂಗಲದ ಬಳಿ."
        ),
        "language": "kn",
        "expected_category": "Garbage / Sanitation",
    },
    {
        "name": "HI → Garbage / Sanitation (कचरा समस्या)",
        "text": (
            "पांच दिनों से कचरा उठाया नहीं गया है। "
            "कचरे का ढेर लग गया है। कचरा गाड़ी नहीं आ रही। "
            "सफाई करवाओ कचरा उठाओ।"
        ),
        "language": "hi",
        "expected_category": "Garbage / Sanitation",
    },

    # ── WATER SUPPLY ──────────────────────────────────────────────────────────
    {
        "name": "EN → Water Supply (no water)",
        "text": (
            "There is no water supply in our apartment building since yesterday. "
            "The tap is completely dry. Water pipeline issue. "
            "We need water supply restored urgently."
        ),
        "language": "en",
        "expected_category": "Water Supply",
    },
    {
        "name": "KN → Water Supply (ನೀರು ಸರಬರಾಜು)",
        "text": (
            "ನಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ನೀರು ಸರಬರಾಜು ಇಲ್ಲ. "
            "ನಲ್ಲಿಯಲ್ಲಿ ನೀರು ಬರುತ್ತಿಲ್ಲ. ಪೈಪ್‌ಲೈನ್ ಸಮಸ್ಯೆ. "
            "ನೀರು ಸರಬರಾಜು ಪುನಃ ಪ್ರಾರಂಭಿಸಿ."
        ),
        "language": "kn",
        "expected_category": "Water Supply",
    },
    {
        "name": "HI → Water Supply (पानी की समस्या)",
        "text": (
            "हमारे इलाके में पानी की आपूर्ति नहीं हो रही है। "
            "नल में पानी नहीं आ रहा है। पाइपलाइन में समस्या है। "
            "पानी सप्लाई चालू करो।"
        ),
        "language": "hi",
        "expected_category": "Water Supply",
    },

    # ── DRAINAGE / SWD ────────────────────────────────────────────────────────
    {
        "name": "EN → Drainage / SWD (blocked drain)",
        "text": (
            "The drainage is completely blocked and overflowing onto the road near Jayanagar. "
            "Sewage water is stagnating. Storm water drain clogged. "
            "Drainage overflow causing flooding on the street."
        ),
        "language": "en",
        "expected_category": "Drainage / SWD",
    },
    {
        "name": "KN → Drainage / SWD (ಚರಂಡಿ ಬ್ಲಾಕ್)",
        "text": (
            "ಜಯನಗರದ ಬಳಿ ಚರಂಡಿ ಸಂಪೂರ್ಣವಾಗಿ ಬ್ಲಾಕ್ ಆಗಿ ರಸ್ತೆಗೆ ಉಕ್ಕಿ ಹರಿಯುತ್ತಿದೆ. "
            "ಒಳಚರಂಡಿ ನೀರು ನಿಂತಿದೆ. ಮಳೆನೀರು ಚರಂಡಿ ಕಟ್ಟಿಕೊಂಡಿದೆ. "
            "ಚರಂಡಿ ಉಕ್ಕಿ ಹರಿದು ರಸ್ತೆಯಲ್ಲಿ ನೀರು ನಿಂತಿದೆ."
        ),
        "language": "kn",
        "expected_category": "Drainage / SWD",
    },
    {
        "name": "HI → Drainage / SWD (नाली बंद)",
        "text": (
            "जयनगर के पास नाली पूरी तरह से बंद हो गई है और सड़क पर बह रही है। "
            "सीवेज का पानी जमा हो गया है। बरसाती नाला बंद है। "
            "नाली ओवरफ्लो होने से सड़क पर पानी भर गया है।"
        ),
        "language": "hi",
        "expected_category": "Drainage / SWD",
    },

    # ── HEALTH / SANITATION ───────────────────────────────────────────────────
    {
        "name": "EN → Health / Sanitation (mosquito breeding)",
        "text": (
            "There is heavy mosquito breeding near the stagnant water pond in our locality. "
            "Dengue and malaria cases have been reported nearby. "
            "Health department fumigation and sanitation drive is urgently needed."
        ),
        "language": "en",
        "expected_category": "Health / Sanitation",
    },
    {
        "name": "KN → Health / Sanitation (ಸೊಳ್ಳೆ ಡೆಂಗ್ಯೂ)",
        "text": (
            "ನಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ನಿಂತ ನೀರಿನಿಂದ ಸೊಳ್ಳೆಗಳ ಸಂತಾನೋತ್ಪತ್ತಿ ಹೆಚ್ಚಾಗಿದೆ. "
            "ಡೆಂಗ್ಯೂ ಮಲೇರಿಯಾ ಪ್ರಕರಣಗಳು ವರದಿಯಾಗಿವೆ. "
            "ಆರೋಗ್ಯ ಇಲಾಖೆ ಫಾಗಿಂಗ್ ಮತ್ತು ಸ್ವಚ್ಛತಾ ಕಾರ್ಯಕ್ರಮ ತುರ್ತಾಗಿ ಬೇಕು."
        ),
        "language": "kn",
        "expected_category": "Health / Sanitation",
    },
    {
        "name": "HI → Health / Sanitation (मच्छर डेंगू)",
        "text": (
            "हमारे इलाके में रुके हुए पानी से मच्छरों का प्रकोप बहुत बढ़ गया है। "
            "डेंगू और मलेरिया के मामले सामने आए हैं। "
            "स्वास्थ्य विभाग द्वारा फॉगिंग और सफाई अभियान तुरंत जरूरी है।"
        ),
        "language": "hi",
        "expected_category": "Health / Sanitation",
    },

    # ── PARKS ─────────────────────────────────────────────────────────────────
    {
        "name": "EN → Parks (broken playground)",
        "text": (
            "The children's playground equipment in our neighborhood park is broken and rusted, "
            "posing a safety hazard. Trees inside the park garden have not been maintained. "
            "Parks department needs to repair the swings and slides."
        ),
        "language": "en",
        "expected_category": "Parks",
    },
    {
        "name": "KN → Parks (ಉದ್ಯಾನವನ ಹಾಳಾಗಿದೆ)",
        "text": (
            "ನಮ್ಮ ಬಡಾವಣೆಯ ಉದ್ಯಾನವನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟದ ಸಾಮಗ್ರಿಗಳು ಮುರಿದು ತುಕ್ಕು ಹಿಡಿದಿವೆ, "
            "ಅಪಾಯಕಾರಿಯಾಗಿದೆ. ಉದ್ಯಾನದ ಮರಗಳ ನಿರ್ವಹಣೆ ಆಗಿಲ್ಲ. "
            "ಪಾರ್ಕ್ಸ್ ಇಲಾಖೆ ಜೋಕಾಲಿ ಮತ್ತು ಸ್ಲೈಡ್ ರಿಪೇರಿ ಮಾಡಬೇಕು."
        ),
        "language": "kn",
        "expected_category": "Parks",
    },
    {
        "name": "HI → Parks (पार्क टूटा हुआ)",
        "text": (
            "हमारे मोहल्ले के पार्क में बच्चों के खेलने के उपकरण टूट गए हैं और जंग लग गया है, "
            "यह खतरनाक है। बगीचे के पेड़ों का रखरखाव नहीं हुआ है। "
            "पार्क विभाग को झूले और स्लाइड की मरम्मत करनी चाहिए।"
        ),
        "language": "hi",
        "expected_category": "Parks",
    },

    # ── TRAFFIC ───────────────────────────────────────────────────────────────
    {
        "name": "EN → Traffic (signal malfunction)",
        "text": (
            "The traffic signal at the main junction near Marathahalli has been malfunctioning "
            "for a week, causing severe traffic jams and near-accidents. "
            "Traffic engineering cell needs to fix the signal urgently."
        ),
        "language": "en",
        "expected_category": "Traffic",
    },
    {
        "name": "KN → Traffic (ಟ್ರಾಫಿಕ್ ಸಿಗ್ನಲ್)",
        "text": (
            "ಮಾರತಹಳ್ಳಿ ಬಳಿಯ ಪ್ರಮುಖ ಜಂಕ್ಷನ್‌ನಲ್ಲಿ ಟ್ರಾಫಿಕ್ ಸಿಗ್ನಲ್ ಒಂದು ವಾರದಿಂದ ಕೆಟ್ಟಿದೆ, "
            "ಇದರಿಂದ ವಾಹನ ದಟ್ಟಣೆ ಮತ್ತು ಅಪಘಾತದ ಅಪಾಯ ಉಂಟಾಗುತ್ತಿದೆ. "
            "ಟ್ರಾಫಿಕ್ ಎಂಜಿನಿಯರಿಂಗ್ ಸೆಲ್ ಸಿಗ್ನಲ್ ದುರಸ್ತಿ ಮಾಡಬೇಕು."
        ),
        "language": "kn",
        "expected_category": "Traffic",
    },
    {
        "name": "HI → Traffic (ट्रैफिक सिग्नल खराब)",
        "text": (
            "मराठाहल्ली के पास मुख्य चौराहे पर ट्रैफिक सिग्नल एक हफ्ते से खराब है, "
            "जिससे भारी जाम और दुर्घटना का खतरा बना हुआ है। "
            "ट्रैफिक इंजीनियरिंग सेल को सिग्नल तुरंत ठीक करना चाहिए।"
        ),
        "language": "hi",
        "expected_category": "Traffic",
    },

    # ── TOWN PLANNING ─────────────────────────────────────────────────────────
    {
        "name": "EN → Town Planning (unauthorized construction)",
        "text": (
            "An unauthorized commercial building is being constructed in a residential zone "
            "near our street without proper approval. This violates zoning regulations. "
            "Town planning department should inspect and take action."
        ),
        "language": "en",
        "expected_category": "Town Planning",
    },
    {
        "name": "KN → Town Planning (ಅನಧಿಕೃತ ಕಟ್ಟಡ)",
        "text": (
            "ನಮ್ಮ ಬೀದಿಯ ಬಳಿ ವಸತಿ ವಲಯದಲ್ಲಿ ಅನಧಿಕೃತ ವಾಣಿಜ್ಯ ಕಟ್ಟಡ ಸರಿಯಾದ ಅನುಮತಿ ಇಲ್ಲದೆ "
            "ನಿರ್ಮಾಣವಾಗುತ್ತಿದೆ. ಇದು ವಲಯ ನಿಯಮಗಳ ಉಲ್ಲಂಘನೆ. "
            "ನಗರ ಯೋಜನಾ ಇಲಾಖೆ ಪರಿಶೀಲಿಸಿ ಕ್ರಮ ಕೈಗೊಳ್ಳಬೇಕು."
        ),
        "language": "kn",
        "expected_category": "Town Planning",
    },
    {
        "name": "HI → Town Planning (अवैध निर्माण)",
        "text": (
            "हमारी गली के पास आवासीय क्षेत्र में बिना उचित अनुमति के एक अवैध व्यावसायिक इमारत "
            "बनाई जा रही है। यह जोनिंग नियमों का उल्लंघन है। "
            "नगर योजना विभाग को निरीक्षण कर कार्रवाई करनी चाहिए।"
        ),
        "language": "hi",
        "expected_category": "Town Planning",
    },

    # ── REVENUE ───────────────────────────────────────────────────────────────
    {
        "name": "EN → Revenue (khata transfer pending)",
        "text": (
            "I have applied for a property tax khata transfer three months ago but there has "
            "been no update or response from the revenue department. "
            "The property documents and khata certificate are pending."
        ),
        "language": "en",
        "expected_category": "Revenue",
    },
    {
        "name": "KN → Revenue (ಖಾತಾ ವರ್ಗಾವಣೆ)",
        "text": (
            "ನಾನು ಮೂರು ತಿಂಗಳ ಹಿಂದೆ ಆಸ್ತಿ ತೆರಿಗೆ ಖಾತಾ ವರ್ಗಾವಣೆಗೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಿದ್ದೇನೆ ಆದರೆ "
            "ಕಂದಾಯ ಇಲಾಖೆಯಿಂದ ಯಾವುದೇ ಅಪ್‌ಡೇಟ್ ಬಂದಿಲ್ಲ. "
            "ಆಸ್ತಿ ದಾಖಲೆಗಳು ಮತ್ತು ಖಾತಾ ಪ್ರಮಾಣಪತ್ರ ಬಾಕಿ ಇವೆ."
        ),
        "language": "kn",
        "expected_category": "Revenue",
    },
    {
        "name": "HI → Revenue (खाता ट्रांसफर लंबित)",
        "text": (
            "मैंने तीन महीने पहले प्रॉपर्टी टैक्स खाता ट्रांसफर के लिए आवेदन किया था लेकिन "
            "राजस्व विभाग से कोई अपडेट नहीं मिला। "
            "संपत्ति के दस्तावेज़ और खाता प्रमाणपत्र लंबित हैं।"
        ),
        "language": "hi",
        "expected_category": "Revenue",
    },

    # ── VETERINARY ────────────────────────────────────────────────────────────
    {
        "name": "EN → Veterinary (stray dogs)",
        "text": (
            "There are several stray dogs in our locality that appear injured and unvaccinated, "
            "posing a risk to residents. "
            "Veterinary department should conduct sterilization and vaccination camp urgently."
        ),
        "language": "en",
        "expected_category": "Veterinary",
    },
    {
        "name": "KN → Veterinary (ಬೀದಿ ನಾಯಿಗಳು)",
        "text": (
            "ನಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ಗಾಯಗೊಂಡ ಮತ್ತು ಲಸಿಕೆ ಹಾಕಿಸದ ಬೀದಿ ನಾಯಿಗಳು ಹಲವಾರು ಇವೆ, "
            "ಇದು ನಿವಾಸಿಗಳಿಗೆ ಅಪಾಯಕಾರಿ. "
            "ಪಶುವೈದ್ಯಕೀಯ ಇಲಾಖೆ ಸಂತಾನಹರಣ ಮತ್ತು ಲಸಿಕೆ ಶಿಬಿರ ತುರ್ತಾಗಿ ನಡೆಸಬೇಕು."
        ),
        "language": "kn",
        "expected_category": "Veterinary",
    },
    {
        "name": "HI → Veterinary (आवारा कुत्ते)",
        "text": (
            "हमारे इलाके में कई आवारा कुत्ते घायल और बिना टीकाकरण के घूम रहे हैं, "
            "जो निवासियों के लिए खतरा है। "
            "पशु चिकित्सा विभाग को तुरंत नसबंदी और टीकाकरण शिविर आयोजित करना चाहिए।"
        ),
        "language": "hi",
        "expected_category": "Veterinary",
    },

    # ── ADVERTISEMENT ─────────────────────────────────────────────────────────
    {
        "name": "EN → Advertisement (illegal hoarding)",
        "text": (
            "Illegal advertisement hoardings and flex banners have been put up without permission "
            "along the main road, blocking visibility for drivers. "
            "Advertisement department should remove these unauthorized banners."
        ),
        "language": "en",
        "expected_category": "Advertisement",
    },
    {
        "name": "KN → Advertisement (ಅಕ್ರಮ ಹೋರ್ಡಿಂಗ್)",
        "text": (
            "ಮುಖ್ಯ ರಸ್ತೆಯ ಉದ್ದಕ್ಕೂ ಅನುಮತಿ ಇಲ್ಲದೆ ಅಕ್ರಮ ಜಾಹೀರಾತು ಹೋರ್ಡಿಂಗ್ ಮತ್ತು ಫ್ಲೆಕ್ಸ್ "
            "ಬ್ಯಾನರ್‌ಗಳನ್ನು ಹಾಕಲಾಗಿದೆ, ಇದು ಚಾಲಕರ ದೃಷ್ಟಿಗೆ ಅಡ್ಡಿಯಾಗುತ್ತಿದೆ. "
            "ಜಾಹೀರಾತು ಇಲಾಖೆ ಈ ಅನಧಿಕೃತ ಬ್ಯಾನರ್‌ಗಳನ್ನು ತೆಗೆದುಹಾಕಬೇಕು."
        ),
        "language": "kn",
        "expected_category": "Advertisement",
    },
    {
        "name": "HI → Advertisement (अवैध होर्डिंग)",
        "text": (
            "मुख्य सड़क के किनारे बिना अनुमति के अवैध विज्ञापन होर्डिंग और फ्लेक्स बैनर लगाए गए हैं, "
            "जिससे चालकों की दृश्यता बाधित हो रही है। "
            "विज्ञापन विभाग को इन अनधिकृत बैनरों को हटाना चाहिए।"
        ),
        "language": "hi",
        "expected_category": "Advertisement",
    },

    # ── OTHERS ────────────────────────────────────────────────────────────────
    {
        "name": "EN → Others (general civic issue)",
        "text": (
            "There is a general civic maintenance issue in our area that does not fit standard "
            "categories - a broken compound wall of a public building is collapsing and needs "
            "immediate attention from the concerned department."
        ),
        "language": "en",
        "expected_category": "Others",
    },
    {
        "name": "KN → Others (ಸಾಮಾನ್ಯ ಸಮಸ್ಯೆ)",
        "text": (
            "ನಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ಸಾಮಾನ್ಯ ನಾಗರಿಕ ನಿರ್ವಹಣಾ ಸಮಸ್ಯೆ ಇದೆ - ಸಾರ್ವಜನಿಕ ಕಟ್ಟಡದ ಮುರಿದ "
            "ಕಾಂಪೌಂಡ್ ಗೋಡೆ ಕುಸಿಯುತ್ತಿದೆ ಮತ್ತು ಸಂಬಂಧಿಸಿದ ಇಲಾಖೆಯ ತಕ್ಷಣದ ಗಮನ ಬೇಕಾಗಿದೆ."
        ),
        "language": "kn",
        "expected_category": "Others",
    },
    {
        "name": "HI → Others (सामान्य समस्या)",
        "text": (
            "हमारे क्षेत्र में एक सामान्य नागरिक रखरखाव की समस्या है - सार्वजनिक भवन की टूटी हुई "
            "सीमा दीवार ढह रही है और संबंधित विभाग का तुरंत ध्यान चाहिए।"
        ),
        "language": "hi",
        "expected_category": "Others",
    },

    # ── NON-CIVIC (should be rejected, all 3 languages) ──────────────────────
    {
        "name": "EN → Non-Civic (should reject)",
        "text": (
            "I want to ask about the latest cricket match score and the best "
            "restaurant near my house. What is the weather today? Tell me a joke."
        ),
        "language": "en",
        "expected_category": "Non-Civic",
        "expect_rejection": True,
    },
    {
        "name": "KN → Non-Civic (should reject)",
        "text": (
            "ಇಂದಿನ ಹವಾಮಾನ ಹೇಗಿದೆ? ಕ್ರಿಕೆಟ್ ಪಂದ್ಯದ ಸ್ಕೋರ್ ಏನು? "
            "ಒಳ್ಳೆಯ ಹೋಟೆಲ್ ಎಲ್ಲಿದೆ? ನನಗೆ ಒಂದು ಜೋಕ್ ಹೇಳಿ."
        ),
        "language": "kn",
        "expected_category": "Non-Civic",
        "expect_rejection": True,
    },
    {
        "name": "HI → Non-Civic (should reject)",
        "text": (
            "आज मौसम कैसा है? क्रिकेट मैच का स्कोर क्या है? "
            "अच्छा रेस्टोरेंट कहां है? मुझे एक जोक सुनाओ।"
        ),
        "language": "hi",
        "expected_category": "Non-Civic",
        "expect_rejection": True,
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ts():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _c(text, code):
    return f"\033[{code}m{text}\033[0m"

def PASS(msg):  return _c(f"[PASS] {msg}", "32")
def FAIL(msg):  return _c(f"[FAIL] {msg}", "31")
def WARN(msg):  return _c(f"[WARN] {msg}", "33")
def INFO(msg):  return _c(f"[INFO] {msg}", "36")
def BOLD(msg):  return _c(msg, "1")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Verify ML model is loaded and active
# ═══════════════════════════════════════════════════════════════════════════════

def verify_ml_model_loaded():
    """Check /model/status to confirm the trained ML model is loaded."""
    print(f"\n{'='*72}")
    print(BOLD("  SECTION 1: ML MODEL VERIFICATION"))
    print(f"{'='*72}\n")

    res = requests.get(f"{BASE_URL}/model/status", timeout=120)
    if res.status_code != 200:
        print(FAIL(f"/model/status returned HTTP {res.status_code}"))
        return False

    data = res.json()
    model_path    = data.get("path", "NOT FOUND")
    class_count   = data.get("class_count", 0)
    vocab_size    = data.get("vocabulary_size", 0)
    classes       = data.get("classes", [])
    zs            = data.get("zero_shot_fallback", {})

    checks = []

    # Check 1: Model file exists
    model_exists = bool(model_path and class_count > 0)
    checks.append(model_exists)
    status = PASS if model_exists else FAIL
    print(status(f"ML model loaded from: {model_path}"))

    # Check 2: Has correct number of categories
    has_classes = class_count >= 10
    checks.append(has_classes)
    status = PASS if has_classes else FAIL
    print(status(f"Model has {class_count} categories: {', '.join(classes[:5])}..."))

    # Check 3: Vocabulary is non-trivial
    has_vocab = vocab_size >= 100
    checks.append(has_vocab)
    status = PASS if has_vocab else FAIL
    print(status(f"TF-IDF vocabulary: {vocab_size} tokens"))

    # Check 4: Zero-shot fallback info
    zs_loaded = zs.get("loaded", False)
    status = PASS if zs_loaded else INFO
    print(status(f"Zero-shot fallback: {zs.get('model')} (loaded={zs_loaded})"))

    all_ok = all(checks)
    if all_ok:
        print(f"\n  {PASS('ML model is active and ready for predictions')}")
    else:
        print(f"\n  {FAIL('ML model verification failed')}")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Direct ML model prediction (offline, no HTTP)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_ml_model_direct():
    """Load the pkl directly and verify predictions come from the trained model."""
    print(f"\n{'='*72}")
    print(BOLD("  SECTION 2: DIRECT ML MODEL PREDICTION (No HTTP)"))
    print(f"{'='*72}\n")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        import pickle
        from nlp_features import build_multilingual_classification_text
    except ImportError as e:
        print(WARN(f"Cannot import required modules: {e}"))
        return 0, 0

    # Find model file
    candidates = ["Models/model_bbmp.pkl", "model_bbmp.pkl"]
    pkg = None
    for path in candidates:
        try:
            with open(path, "rb") as f:
                pkg = pickle.load(f)
            print(INFO(f"Loaded model from: {path}"))
            break
        except FileNotFoundError:
            continue

    if pkg is None:
        print(WARN("Could not find model .pkl file"))
        return 0, 0

    vectorizer = pkg["vectorizer"]
    clf = pkg["classifier"]
    classes = list(clf.classes_)

    # Confirm it's a real Naive Bayes classifier
    model_type = type(clf).__name__
    print(INFO(f"Classifier type: {model_type}"))
    print(INFO(f"Classes: {classes}"))
    print(INFO(f"Feature count: {len(vectorizer.get_feature_names_out())}"))

    # ─── Test predictions — one per civic category, plus Non-Civic ──────────
    direct_tests = [
        ("Road has potholes repair urgently road damage", "Road Repair"),
        ("Garbage collection not done waste piled up", "Garbage / Sanitation"),
        ("Street light not working dark electrical", "Street Light"),
        ("Water supply no water tap dry pipeline", "Water Supply"),
        ("Drainage blocked overflow sewage storm water drain", "Drainage / SWD"),
        ("Mosquito dengue breeding health sanitation fogging", "Health / Sanitation"),
        ("Park playground broken equipment tree garden swings", "Parks"),
        ("Traffic signal malfunction junction vehicles jam", "Traffic"),
        ("Unauthorized building construction zoning approval violation", "Town Planning"),
        ("Property tax khata transfer certificate revenue department", "Revenue"),
        ("Stray dogs injured unvaccinated sterilization vaccination camp", "Veterinary"),
        ("Advertisement hoarding illegal banner permission flex", "Advertisement"),
        ("Compound wall collapsing public building general maintenance", "Others"),
        ("Cricket weather restaurant joke irrelevant", "Non-Civic"),
    ]

    passed = 0
    total = len(direct_tests)

    print()
    for i, (text, expected) in enumerate(direct_tests, 1):
        classification_text = build_multilingual_classification_text(text, text)
        vec = vectorizer.transform([classification_text])
        predicted = str(clf.predict(vec)[0])

        # Get confidence
        conf = 0.0
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(vec)[0]
            idx = classes.index(predicted) if predicted in classes else 0
            conf = proba[idx]

        ok = predicted == expected
        if ok:
            passed += 1

        status = PASS if ok else FAIL
        print(status(
            f"[{i:02d}] '{text[:50]}...' → {predicted} (conf={conf:.3f}) | expected: {expected}"
        ))

    return passed, total


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Multi-language API tests (English, Kannada, Hindi)
# ═══════════════════════════════════════════════════════════════════════════════

def run_api_test(tc, idx, total):
    """Submit a complaint via API and verify category + method used."""
    name     = tc["name"]
    text     = tc["text"]
    expected = tc["expected_category"]
    lang     = tc.get("language", "en")
    expect_rejection = tc.get("expect_rejection", False)

    lat = TEST_LATITUDE  + idx * 0.01
    lon = TEST_LONGITUDE + idx * 0.01

    lang_flag = {"en": "🇬🇧", "kn": "🇮🇳", "hi": "🇮🇳"}.get(lang, "🌐")
    lang_name = {"en": "English", "kn": "Kannada", "hi": "Hindi"}.get(lang, lang)

    print(f"\n{'─'*72}")
    print(f"  Test {idx}/{total}: {lang_flag} {name}")
    print(f"  Language          : {lang_name} ({lang})")
    print(f"  Expected category : {expected}")
    print(f"  Text              : {text[:80]}...")

    payload = {
        "live_latitude": lat,
        "live_longitude": lon,
        "live_location_timestamp": _ts(),
        "text_note": text,
        "language": lang,
        "target_language": "en",
    }

    try:
        res = requests.post(f"{BASE_URL}/submit-complaint", data=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        print(FAIL(f"Cannot connect to backend at {BASE_URL}"))
        return False, {}
    except requests.exceptions.Timeout:
        print(FAIL("Request timed out after 120 seconds"))
        return False, {}

    # Non-Civic: expect HTTP 400 rejection
    if expect_rejection:
        if res.status_code == 400:
            print(PASS("Correctly rejected as Non-Civic (HTTP 400)"))
            return True, {"method": "rejection", "category": "Non-Civic", "language": lang}
        else:
            print(FAIL(f"Expected HTTP 400, got {res.status_code}"))
            return False, {}

    # Normal complaint: expect HTTP 200
    if res.status_code != 200:
        print(FAIL(f"HTTP {res.status_code} | {res.text[:200]}"))
        return False, {}

    body = res.json()
    actual_category = body.get("category", "")
    explanation     = body.get("category_explanation", {})
    method          = explanation.get("method", "unknown")
    confidence      = explanation.get("confidence", "N/A")
    top_features    = explanation.get("top_features", [])
    highlight_terms = explanation.get("highlight_terms", [])
    transcribed     = body.get("transcribed_text", "")[:100]
    translated      = body.get("translated_text", "")[:100]
    detected_lang   = body.get("detected_language", "")
    is_dup          = body.get("duplicate", False)

    print(f"  Actual category   : {actual_category}")
    print(f"  Confidence        : {confidence}")
    print(f"  ML Method         : {method}")
    print(f"  Detected language : {detected_lang}")

    if lang != "en":
        print(f"  Transcribed       : {transcribed}...")
        print(f"  Translated (EN)   : {translated}...")

    if top_features:
        feature_strs = [f"{f['term']} ({f['importance_percent']}%)" for f in top_features[:5]]
        print(f"  Top features      : {', '.join(feature_strs)}")

        # Verify no char_wb__ junk appears
        has_char_wb = any("char_wb__" in f["term"] for f in top_features)
        if has_char_wb:
            print(WARN("  ⚠ char_wb__ character n-grams still appearing in features!"))
    else:
        print(f"  Highlight terms   : {highlight_terms}")

    if is_dup:
        print(f"  [Note] Matched existing complaint #{body.get('id')} (votes={body.get('votes')})")

    passed = actual_category == expected
    if passed:
        print(PASS(f"Category matches '{expected}'" + (" [duplicate]" if is_dup else " [fresh]")))
    else:
        print(FAIL(f"Expected '{expected}', got '{actual_category}'"))

    return passed, {
        "method": method,
        "confidence": confidence,
        "category": actual_category,
        "language": lang,
        "features": [f["term"] for f in top_features[:5]] if top_features else highlight_terms,
    }


def run_all_api_tests():
    """Run all multi-language, all-category API tests."""
    print(f"\n{'='*72}")
    print(BOLD("  SECTION 3: MULTI-LANGUAGE, ALL-CATEGORY API TESTS"))
    print(BOLD(f"  {len(TEST_CASES)} tests | EN + KN + HI | Backend: {BASE_URL}"))
    print(f"{'='*72}")

    results = []
    details = []

    for i, tc in enumerate(TEST_CASES, 1):
        passed, info = run_api_test(tc, i, len(TEST_CASES))
        results.append(passed)
        details.append({"test": tc["name"], "passed": passed, "category": tc["expected_category"], **info})
        time.sleep(0.5)

    return results, details


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print(BOLD("  CIVIC COMPLAINT — ML MODEL & MULTI-LANGUAGE TEST SUITE"))
    print(f"  GPS base: ({TEST_LATITUDE:.4f}, {TEST_LONGITUDE:.4f}) [seed={_run_seed}]")
    print("=" * 72)

    # ── Section 1: Verify ML model is loaded ──────────────────────────────────
    model_ok = verify_ml_model_loaded()
    if not model_ok:
        print(FAIL("ML model not loaded. Cannot proceed."))
        sys.exit(1)

    # ── Section 2: Direct model verification ──────────────────────────────────
    direct_passed, direct_total = verify_ml_model_direct()

    # ── Section 3: Multi-language API tests ───────────────────────────────────
    api_results, api_details = run_all_api_tests()
    api_passed = sum(api_results)
    api_total  = len(api_results)

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(BOLD("  TEST SUMMARY"))
    print(f"{'='*72}\n")

    # Section 1
    print(f"  ML Model Loaded   : {PASS('YES') if model_ok else FAIL('NO')}")

    # Section 2
    if direct_total > 0:
        s2_ok = direct_passed == direct_total
        print(f"  Direct Prediction : {direct_passed}/{direct_total} passed  "
              f"{'OK' if s2_ok else 'FAIL'}")

    # Section 3 — by language
    en_tests = [d for d in api_details if d.get("language") == "en" and d.get("category") != "Non-Civic"]
    kn_tests = [d for d in api_details if d.get("language") == "kn" and d.get("category") != "Non-Civic"]
    hi_tests = [d for d in api_details if d.get("language") == "hi" and d.get("category") != "Non-Civic"]
    reject_tests = [d for d in api_details if d.get("method") == "rejection"]

    en_pass = sum(1 for d in en_tests if d["passed"])
    kn_pass = sum(1 for d in kn_tests if d["passed"])
    hi_pass = sum(1 for d in hi_tests if d["passed"])
    reject_pass = sum(1 for d in reject_tests if d["passed"])

    print(f"\n  API Tests by Language:")
    print(f"    🇬🇧 English       : {en_pass}/{len(en_tests)} passed")
    print(f"    🇮🇳 Kannada       : {kn_pass}/{len(kn_tests)} passed")
    print(f"    🇮🇳 Hindi         : {hi_pass}/{len(hi_tests)} passed")
    print(f"    🚫 Non-Civic      : {reject_pass}/{len(reject_tests)} rejected")
    print(f"    ─────────────────────────")
    print(f"    Total API Tests   : {api_passed}/{api_total} passed")

    # Section 3 — by category
    categories = sorted(set(d.get("category") for d in api_details if d.get("category") != "Non-Civic"))
    print(f"\n  API Tests by Category:")
    for cat in categories:
        cat_tests = [d for d in api_details if d.get("category") == cat]
        cat_pass = sum(1 for d in cat_tests if d["passed"])
        mark = "✅" if cat_pass == len(cat_tests) else "❌"
        print(f"    {mark} {cat:<24}: {cat_pass}/{len(cat_tests)} passed")

    # Verify ML model is used (not just zero-shot)
    ml_method_tests = [d for d in api_details if d.get("method") == "tfidf_multinomial_nb"]
    zs_method_tests = [d for d in api_details if d.get("method") == "zero_shot_nli_fallback"]

    print(f"\n  Classification Method Breakdown:")
    print(f"    TF-IDF + NB (ML) : {len(ml_method_tests)} complaints")
    print(f"    Zero-shot (NLI)  : {len(zs_method_tests)} complaints")
    print(f"    Rejection        : {len(reject_tests)} complaints")

    if len(ml_method_tests) > 0:
        print(f"\n  {PASS('CONFIRMED: Your trained ML model (TF-IDF + Naive Bayes) IS being used for predictions!')}")
    else:
        print(f"\n  {WARN('WARNING: All predictions used zero-shot fallback. ML model may not be effective.')}")

    # Feature cleanliness check
    has_char_wb = False
    for d in api_details:
        for feat in d.get("features", []):
            if "char_wb__" in str(feat):
                has_char_wb = True
                break

    if not has_char_wb:
        print(f"  {PASS('Feature explanations are clean (no char_wb__ junk)')}")
    else:
        print(f"  {FAIL('char_wb__ tokens still appearing in feature explanations')}")

    # Overall result
    all_pass = (
        model_ok
        and (direct_passed == direct_total)
        and (api_passed == api_total)
    )
    colour = "32" if all_pass else "31"
    overall = "ALL TESTS PASSED ✅" if all_pass else "SOME TESTS FAILED ❌"
    print(f"\n  {_c(overall, colour)}\n")
    print("=" * 72)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)