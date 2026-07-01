"""
test_classification.py
======================
10 end-to-end classification tests that verify the HuggingFace-connected
TF-IDF + Naive Bayes classifier (with zero-shot NLI fallback) correctly
routes each complaint to the expected civic category.

Tests hit the live backend at http://localhost:8000/submit-complaint via
text_note (no audio required) so they run without mic/file dependencies.

Run:
    python tests/test_classification.py
"""

import requests
import json
import time
import sys
import os
from datetime import datetime, timezone

# Force UTF-8 output on Windows terminals (avoids cp1252 encode errors)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# 10 Diverse complaint examples: (description, text, expected_category)
# Expected categories must be one of the 15 model classes.
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id": 1,
        "name": "Road Repair - Pothole complaint",
        "text": (
            "The road has potholes which need to be repaired urgently. "
            "Road repair complaint near Koramangala junction — several big potholes "
            "on the street. Two-wheelers are skidding due to road damage. "
            "Urgent road repair work required on this road surface."
        ),
        "expected_category": "Road Repair",
        "language": "en",
    },
    {
        "id": 2,
        "name": "Garbage / Sanitation - Waste not collected",
        "text": (
            "Garbage collection not done. Waste piled up. "
            "Garbage bin overflow. Collection missed. "
            "Garbage truck not coming. BTM Layout 2nd stage."
        ),
        "expected_category": "Garbage / Sanitation",
        "language": "en",
    },
    {
        "id": 3,
        "name": "Street Light — Lamp not working",
        "text": (
            "The street light on 80 feet road near Indiranagar has not been working "
            "for three weeks. The entire street is dark at night. There is no lamp "
            "light on this road and it is very unsafe. Please fix the electrical issue."
        ),
        "expected_category": "Street Light",
        "language": "en",
    },
    {
        "id": 4,
        "name": "Water Supply — No water from tap",
        "text": (
            "There is no water supply in our apartment building since yesterday. "
            "The tap is completely dry. Our water connection is dead and we are "
            "struggling for drinking water. Pipeline seems broken. Please restore "
            "the water supply immediately."
        ),
        "expected_category": "Water Supply",
        "language": "en",
    },
    {
        "id": 5,
        "name": "Drainage / SWD — Overflow and flooding",
        "text": (
            "The drainage is completely blocked and overflowing onto the road near "
            "Jayanagar 4th block. Sewage water is flooding the street. The storm "
            "water drain is clogged and foul water is stagnating. Need immediate "
            "drainage cleaning."
        ),
        "expected_category": "Drainage / SWD",
        "language": "en",
    },
    {
        "id": 6,
        "name": "Health / Sanitation — Mosquito breeding",
        "text": (
            "Stagnant water in open pits near HSR Layout is causing mosquito breeding "
            "and dengue risk. Public health is at stake. There are mosquito larvae "
            "in the water logged area. Sanitation department must fumigate and clean "
            "this area to prevent disease spread."
        ),
        "expected_category": "Health / Sanitation",
        "language": "en",
    },
    {
        "id": 7,
        "name": "Parks - Playground issue",
        "text": (
            "The public park near Whitefield has broken swings and damaged playground "
            "equipment. Children are getting hurt. The park benches are broken and "
            "the garden is not maintained. Trees are not trimmed. Park needs immediate "
            "maintenance and repair."
        ),
        "expected_category": "Parks",
        "language": "en",
    },
    {
        "id": 8,
        "name": "Traffic — Signal not working",
        "text": (
            "The traffic signal at MG Road junction has been malfunctioning for two "
            "days. The signal light is broken and not functioning. There is heavy "
            "traffic congestion because no traffic signals are working at this "
            "intersection. Vehicles are stuck in jam."
        ),
        "expected_category": "Traffic",
        "language": "en",
    },
    {
        "id": 9,
        "name": "Advertisement — Illegal hoarding",
        "text": (
            "There is an illegal advertisement hoarding erected without permission "
            "near Marathahalli bridge. The unauthorized banner is obstructing the "
            "road view and is a safety hazard. Multiple large flex boards are put up "
            "without BBMP approval. Please remove them."
        ),
        "expected_category": "Advertisement",
        "language": "en",
    },
    {
        "id": 10,
        "name": "Non-Civic - Irrelevant complaint (should be rejected)",
        "text": (
            "I want to ask about the latest cricket match score and the best "
            "restaurant near my house. What is the weather today? Tell me a joke."
        ),
        "expected_category": "Non-Civic",  # Backend returns HTTP 400 for Non-Civic
        "language": "en",
        "expect_rejection": True,   # HTTP 400 expected
    },
]

# Bangalore city centre
_BASE_LAT = 12.9716
_BASE_LON = 77.5946

# Shift the origin by a time-based seed so each test run lands in a fresh
# GPS zone that won't collide with complaints from previous runs.
# 0.001 deg ~ 111 m; shifting by 0.02..0.04 deg puts each run ~3-5 km away.
import time as _time
_run_seed = int(_time.time()) % 1000   # 0-999
_RUN_LAT_OFFSET = 0.020 + (_run_seed % 20) * 0.001   # 0.020 ... 0.039
_RUN_LON_OFFSET = 0.020 + (_run_seed // 20) * 0.001  # 0.020 ... 0.069

TEST_LATITUDE  = _BASE_LAT + _RUN_LAT_OFFSET
TEST_LONGITUDE = _BASE_LON + _RUN_LON_OFFSET
print(f"[INFO] Test run GPS base: ({TEST_LATITUDE:.4f}, {TEST_LONGITUDE:.4f}) [seed={_run_seed}]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _color(text: str, code: str) -> str:
    """ANSI colour helper (works in most terminals)."""
    return f"\033[{code}m{text}\033[0m"


def PASS(msg: str) -> str:
    return _color(f"[PASS] {msg}", "32")


def FAIL(msg: str) -> str:
    return _color(f"[FAIL] {msg}", "31")


def WARN(msg: str) -> str:
    return _color(f"[WARN] {msg}", "33")


def INFO(msg: str) -> str:
    return _color(f"[INFO] {msg}", "36")


# ---------------------------------------------------------------------------
# Pre-flight: verify model status endpoint
# ---------------------------------------------------------------------------

def check_model_status() -> dict:
    """Confirm the classifier is connected and zero-shot fallback is loaded."""
    print(INFO("Checking /model/status ..."))
    res = requests.get(f"{BASE_URL}/model/status", timeout=10)
    if res.status_code != 200:
        raise RuntimeError(f"/model/status returned HTTP {res.status_code}: {res.text}")
    data = res.json()
    print(f"  Model path      : {data.get('path')}")
    print(f"  Categories      : {data.get('class_count')} classes")
    print(f"  Vocabulary size : {data.get('vocabulary_size')} tokens")
    zs = data.get("zero_shot_fallback", {})
    print(f"  Zero-shot model : {zs.get('model')} (enabled={zs.get('enabled')}, loaded={zs.get('loaded')})")
    return data


# ---------------------------------------------------------------------------
# Single test runner
# ---------------------------------------------------------------------------

def run_single_test(tc: dict, idx: int, total: int) -> bool:
    """
    Submit a text-based complaint and verify the returned category.
    Returns True if the test passed, False otherwise.
    """
    name = tc["name"]
    text = tc["text"]
    expected = tc["expected_category"]
    expect_rejection = tc.get("expect_rejection", False)

    # Each test steps 0.0005 deg (~55 m) within the run's unique GPS zone so
    # complaints with different categories don't accidentally merge as duplicates
    lat = TEST_LATITUDE  + tc.get("lat_offset", idx * 0.0005)
    lon = TEST_LONGITUDE + tc.get("lon_offset", idx * 0.0005)

    print(f"\n{'-' * 60}")
    print(f"Test {idx}/{total}: {name}")
    print(f"  Expected category : {expected}")
    print(f"  Text snippet      : {text[:80]}...")

    payload = {
        "live_latitude":           lat,
        "live_longitude":          lon,
        "live_location_timestamp": _ts(),
        "text_note":               text,
        "language":                tc.get("language", "en"),
        "target_language":         "en",
    }

    try:
        res = requests.post(
            f"{BASE_URL}/submit-complaint",
            data=payload,
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        print(FAIL(f"Cannot connect to backend at {BASE_URL}. Is uvicorn running?"))
        return False
    except requests.exceptions.Timeout:
        print(FAIL("Request timed out after 60 seconds."))
        return False

    # ── Non-Civic: expect HTTP 400 rejection ─────────────────────────
    if expect_rejection:
        if res.status_code == 400:
            body = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
            detail = body.get("detail", res.text)
            print(PASS(f"Correctly rejected with HTTP 400 | detail: {detail[:80]}"))
            return True
        else:
            print(FAIL(
                f"Expected HTTP 400 rejection for Non-Civic, "
                f"got HTTP {res.status_code} | body: {res.text[:120]}"
            ))
            return False

    # ── Normal civic complaint: expect HTTP 200 ───────────────────────
    if res.status_code != 200:
        print(FAIL(f"HTTP {res.status_code} | body: {res.text[:200]}"))
        return False

    try:
        body = res.json()
    except Exception:
        print(FAIL("Response is not valid JSON"))
        return False

    actual_category = body.get("category", "")
    is_dup = body.get("duplicate", False)

    # Explanation details (key is 'category_explanation' in fresh submissions)
    explanation = body.get("category_explanation", {})
    confidence  = explanation.get("confidence", "N/A (duplicate)" if is_dup else "N/A")
    method      = explanation.get("method", "duplicate_match" if is_dup else "unknown")
    highlights  = explanation.get("highlight_terms", [])

    print(f"  Actual category   : {actual_category}")
    print(f"  Confidence        : {confidence}")
    print(f"  Method            : {method}")
    print(f"  Highlight terms   : {highlights}")
    print(f"  Duplicate?        : {is_dup}")
    if is_dup:
        print(f"  [Note] Matched existing complaint #{body.get('id')} (votes={body.get('votes')})")

    passed = actual_category == expected
    if passed:
        print(PASS(f"Category matches '{expected}'" + (" [via duplicate]" if is_dup else " [fresh submission]")))
    else:
        print(FAIL(f"Expected '{expected}', got '{actual_category}'"))

    return passed


# ---------------------------------------------------------------------------
# Model-level unit tests (no HTTP — uses pickle directly)
# ---------------------------------------------------------------------------

def run_unit_tests() -> list[bool]:
    """
    Directly test the pkl model without HTTP for fast offline verification.
    Returns list of bool results.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        import pickle
        from nlp_features import build_multilingual_classification_text
    except ImportError as e:
        print(WARN(f"Cannot run unit tests (import error): {e}"))
        return []

    # Find the model file
    candidates = ["Models/model_bbmp.pkl", "model_bbmp.pkl"]
    pkg = None
    for path in candidates:
        try:
            with open(path, "rb") as f:
                pkg = pickle.load(f)
            print(INFO(f"Unit tests: loaded model from '{path}'"))
            break
        except FileNotFoundError:
            continue

    if pkg is None:
        print(WARN("Could not find model pkl for unit tests. Skipping."))
        return []

    vectorizer = pkg["vectorizer"]
    clf = pkg["classifier"]
    classes = list(clf.classes_)

    print(f"\n{'='*60}")
    print("UNIT TESTS — Direct model prediction (no HTTP)")
    print(f"{'='*60}")

    unit_cases = [
        ("The road has potholes which need to be repaired. Road repair complaint with several big potholes on the street.",
                                                             "Road Repair"),
        ("garbage collection not done waste piled up garbage bin overflow collection missed garbage truck not coming",
                                                             "Garbage / Sanitation"),
        ("street light lamp not working electrical dark",     "Street Light"),
        ("water supply no water tap dry pipeline",            "Water Supply"),
        ("drainage blocked overflow sewage storm water",      "Drainage / SWD"),
        ("mosquito dengue public health sanitation",          "Health / Sanitation"),
        ("park playground broken tree garden maintenance",    "Parks"),
        ("traffic signal light malfunction junction",         "Traffic"),
        ("advertisement hoarding banner illegal permission",  "Advertisement"),
        ("cricket weather restaurant joke irrelevant",        "Non-Civic"),
    ]

    results = []
    for i, (text, expected) in enumerate(unit_cases, 1):
        classification_text = build_multilingual_classification_text(text, text)
        vec = vectorizer.transform([classification_text])
        predicted = str(clf.predict(vec)[0])

        # Confidence via predict_proba if available
        conf_str = ""
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(vec)[0]
            idx = list(classes).index(predicted) if predicted in classes else 0
            conf_str = f" (conf={proba[idx]:.3f})"

        passed = predicted == expected
        results.append(passed)
        status = PASS if passed else FAIL
        print(status(f"[Unit {i:02d}] '{text[:45]}...' -> {predicted}{conf_str} | expected: {expected}"))

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("=" * 60)
    print("  CIVIC COMPLAINT - CLASSIFICATION TEST SUITE")
    print(f"  {len(TEST_CASES)} Examples | Backend: {BASE_URL}")
    print("=" * 60)

    # ── 0. Unit tests (model pkl, no HTTP) ─────────────────────────
    unit_results = run_unit_tests()
    unit_passed = sum(unit_results)
    unit_total  = len(unit_results)

    # ── 1. Pre-flight model status ──────────────────────────────────
    print(f"\n{'='*60}")
    print("PRE-FLIGHT: Model Status Check")
    print(f"{'='*60}")
    try:
        status_data = check_model_status()
        model_ok = bool(status_data.get("class_count"))
    except Exception as e:
        print(WARN(f"Could not reach backend: {e}"))
        print(WARN("Skipping HTTP tests. Run: uvicorn main:app --port 8000"))
        model_ok = False

    # ── 2. E2E API tests ────────────────────────────────────────────
    api_results = []
    if model_ok:
        print(f"\n{'='*60}")
        print("END-TO-END API TESTS (via /submit-complaint)")
        print(f"{'='*60}")
        for i, tc in enumerate(TEST_CASES, 1):
            result = run_single_test(tc, i, len(TEST_CASES))
            api_results.append(result)
            time.sleep(0.3)   # small delay to avoid rate-limit issues
    else:
        print(WARN("Skipping API tests — backend is not reachable."))
        api_results = [False] * len(TEST_CASES)

    # ── 3. Summary ──────────────────────────────────────────────────
    api_passed = sum(api_results)
    api_total  = len(api_results)

    print(f"\n{'='*60}")
    print("CLASSIFICATION TEST SUMMARY")
    print(f"{'='*60}")

    if unit_total > 0:
        print(f"  Unit Tests   : {unit_passed}/{unit_total} passed"
              f"  {'OK' if unit_passed == unit_total else 'FAIL'}")

    print(f"  API Tests    : {api_passed}/{api_total} passed"
          f"  {'OK' if api_passed == api_total else 'FAIL'}")

    all_pass = (unit_passed == unit_total) and (api_passed == api_total)
    overall  = "ALL TESTS PASSED [OK]" if all_pass else "SOME TESTS FAILED [FAIL]"
    colour   = "32" if all_pass else "31"
    print(_color(f"\n  {overall}\n", colour))

    # Detailed failures
    if api_results and not all(api_results):
        print("  Failed API tests:")
        for i, (passed, tc) in enumerate(zip(api_results, TEST_CASES), 1):
            if not passed:
                print(f"    - [{i}] {tc['name']} (expected: {tc['expected_category']})")

    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
