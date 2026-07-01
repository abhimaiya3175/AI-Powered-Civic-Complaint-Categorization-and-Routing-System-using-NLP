import requests
import json
import time
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def log_test_result(name, passed, detail=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {name} {f'({detail})' if detail else ''}")
    return passed


def run_tests():
    print("=" * 60)
    print("STARTING TEST FOR ALL DATA ANALYTICS & ENERGY METRICS")
    print("=" * 60)
    
    # 1. Login to get JWT Token
    print("\n--- 1. Authenticating Admin User ---")
    try:
        login_res = requests.post(f"{BASE_URL}/login", data={"username": "admin", "password": "change-me-admin-password"})
        if login_res.status_code != 200:
            log_test_result("Admin Login", False, f"Status code: {login_res.status_code}, error: {login_res.text}")
            return
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        log_test_result("Admin Login", True, "Successfully acquired JWT token")
    except Exception as e:
        log_test_result("Admin Login", False, f"Exception: {e}")
        return

    # 2. Fetch Analytics Dashboard Data
    print("\n--- 2. Fetching Analytics Dashboard (No Filters) ---")
    try:
        dashboard_res = requests.get(f"{BASE_URL}/analytics/dashboard", headers=headers)
        if dashboard_res.status_code != 200:
            log_test_result("Fetch Dashboard", False, f"Status code: {dashboard_res.status_code}, error: {dashboard_res.text}")
            return
        data = dashboard_res.json()
        log_test_result("Fetch Dashboard", True, "Successfully retrieved dashboard JSON")
    except Exception as e:
        log_test_result("Fetch Dashboard", False, f"Exception: {e}")
        return

    # 3. Verify Complaint Stats
    print("\n--- 3. Verifying Complaint Statistics ---")
    comp_stats = data.get("complaint_stats", {})
    expected_comp_keys = [
        "total_complaints_processed", "unique_complaints", 
        "duplicate_complaints", "total_votes", "average_votes_per_complaint"
    ]
    all_keys_present = True
    for key in expected_comp_keys:
        val = comp_stats.get(key)
        present = val is not None
        if not present:
            all_keys_present = False
            log_test_result(f"Complaint Stat: {key}", False, "Missing key")
        else:
            log_test_result(f"Complaint Stat: {key}", True, f"Value: {val}")
    log_test_result("Complaint Stats Structure", all_keys_present)

    # 4. Verify NLP Stats
    print("\n--- 4. Verifying NLP Performance Statistics ---")
    nlp_stats = data.get("nlp_stats", {})
    expected_nlp_keys = [
        "total_requests", "avg_processing_time_seconds", "avg_time_by_stage",
        "zero_shot_fallback_rate", "avg_classifier_confidence", "avg_entity_count",
        "entity_type_breakdown", "avg_word_count", "avg_audio_duration"
    ]
    all_nlp_keys_present = True
    for key in expected_nlp_keys:
        val = nlp_stats.get(key)
        present = val is not None
        if not present:
            all_nlp_keys_present = False
            log_test_result(f"NLP Stat: {key}", False, "Missing key")
        else:
            log_test_result(f"NLP Stat: {key}", True, f"Value: {type(val).__name__} = {val}")
    log_test_result("NLP Stats Structure", all_nlp_keys_present)

    # Verify stage breakdown under NLP stats
    avg_time_by_stage = nlp_stats.get("avg_time_by_stage", {})
    expected_stages = ["transcription", "translation", "classification", "ner", "zero_shot"]
    stages_valid = True
    for stage in expected_stages:
        val = avg_time_by_stage.get(stage)
        if val is None or not isinstance(val, (int, float)):
            stages_valid = False
            log_test_result(f"NLP Stage Time: {stage}", False, f"Invalid value: {val}")
        else:
            log_test_result(f"NLP Stage Time: {stage}", True, f"{val}s")
    log_test_result("NLP Stage Times Structure", stages_valid)

    # 5. Verify Energy Stats
    print("\n--- 5. Verifying Energy Analytics ---")
    energy_stats = data.get("energy_stats", {})
    expected_energy_keys = [
        "total_energy_joules", "avg_energy_per_complaint", 
        "energy_saved_by_dedup", "energy_by_stage", "calculation_method"
    ]
    all_energy_keys_present = True
    for key in expected_energy_keys:
        val = energy_stats.get(key)
        present = val is not None
        if not present:
            all_energy_keys_present = False
            log_test_result(f"Energy Stat: {key}", False, "Missing key")
        else:
            log_test_result(f"Energy Stat: {key}", True, f"Value: {val}")
    log_test_result("Energy Stats Structure", all_energy_keys_present)

    # Verify stage breakdown under energy stats
    energy_by_stage = energy_stats.get("energy_by_stage", {})
    energy_stages_valid = True
    for stage in expected_stages:
        val = energy_by_stage.get(stage)
        if val is None or not isinstance(val, (int, float)):
            energy_stages_valid = False
            log_test_result(f"Energy Stage: {stage}", False, f"Invalid value: {val}")
        else:
            log_test_result(f"Energy Stage: {stage}", True, f"{val} J")
    log_test_result("Energy Stage Structure", energy_stages_valid)

    # 6. Verify Error Stats
    print("\n--- 6. Verifying Error Statistics ---")
    error_stats = data.get("error_stats", {})
    expected_err_keys = ["total_errors", "error_rate_percent", "errors_by_stage"]
    all_err_keys_present = True
    for key in expected_err_keys:
        val = error_stats.get(key)
        present = val is not None
        if not present:
            all_err_keys_present = False
            log_test_result(f"Error Stat: {key}", False, "Missing key")
        else:
            log_test_result(f"Error Stat: {key}", True, f"Value: {val}")
    log_test_result("Error Stats Structure", all_err_keys_present)

    # 7. Verify All 15 Charts
    print("\n--- 7. Verifying All 15 Required Charts ---")
    charts = data.get("charts", {})
    expected_charts = [
        ("energy_by_stage", list),
        ("energy_over_time", list),
        ("category_distribution", list),
        ("duplicate_vs_unique", dict),
        ("votes_per_complaint", list),
        ("language_distribution", list),
        ("confidence_histogram", list),
        ("category_language_heatmap", list),
        ("entity_count_histogram", list),
        ("entity_type_breakdown", list),
        ("stage_bottleneck_radar", dict),
        ("throughput_over_time", list),
        ("audio_duration_vs_time", list),
        ("duplicate_cluster_sizes", list),
        ("error_rate_by_stage", list)
    ]
    charts_valid = True
    for chart_name, expected_type in expected_charts:
        chart_data = charts.get(chart_name)
        if chart_data is None:
            charts_valid = False
            log_test_result(f"Chart: {chart_name}", False, "Missing chart data")
        elif not isinstance(chart_data, expected_type):
            charts_valid = False
            log_test_result(f"Chart: {chart_name}", False, f"Incorrect type. Expected {expected_type.__name__}, got {type(chart_data).__name__}")
        else:
            length = len(chart_data) if hasattr(chart_data, "__len__") else "N/A"
            log_test_result(f"Chart: {chart_name}", True, f"Type: {expected_type.__name__}, Size/Keys: {length}")
    log_test_result("All 15 Charts Structure", charts_valid)

    # 8. Verify Data Sources Verification Panel Data
    print("\n--- 8. Verifying Data Source Verification Metadata ---")
    data_sources = data.get("data_sources", {})
    expected_source_keys = [
        "complaint_stats", "nlp_metrics", "energy", "entities", 
        "confidence", "audio_duration", "errors", "note"
    ]
    sources_valid = True
    for key in expected_source_keys:
        val = data_sources.get(key)
        if val is None or not isinstance(val, str):
            sources_valid = False
            log_test_result(f"Data Source: {key}", False, f"Invalid value: {val}")
        else:
            log_test_result(f"Data Source: {key}", True, f"Metadata: '{val[:50]}...'")
    log_test_result("Data Source Verification Structure", sources_valid)

    # 9. Verify Filter Endpoints
    print("\n--- 9. Verifying Dashboard Filters ---")
    # Date filters
    today = datetime.utcnow().strftime("%Y-%m-%d")
    tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    date_filter_query = f"?start_date={today}T00:00:00&end_date={tomorrow}T23:59:59"
    try:
        filtered_res = requests.get(f"{BASE_URL}/analytics/dashboard{date_filter_query}", headers=headers)
        log_test_result("Filter: Date Range", filtered_res.status_code == 200, f"Query: {date_filter_query}")
    except Exception as e:
        log_test_result("Filter: Date Range", False, f"Exception: {e}")

    # Language filter
    lang_filter_query = "?language=english"
    try:
        filtered_res = requests.get(f"{BASE_URL}/analytics/dashboard{lang_filter_query}", headers=headers)
        log_test_result("Filter: Language", filtered_res.status_code == 200, f"Query: {lang_filter_query}")
    except Exception as e:
        log_test_result("Filter: Language", False, f"Exception: {e}")

    # 10. End-to-End Submission & Pipeline Verification
    print("\n--- 10. Submitting New Complaint and Verifying Pipeline Updates ---")
    prev_total = comp_stats.get("total_complaints_processed", 0)
    prev_unique = comp_stats.get("unique_complaints", 0)
    prev_dup = comp_stats.get("duplicate_complaints", 0)

    unique_text = f"Streetlight has been broken for two weeks on 100 feet road. The area is pitch dark and unsafe. Time of test: {datetime.utcnow().isoformat()}"
    test_lat = f"{12.8 + (time.time() % 100) / 1000.0:.6f}"
    test_lon = f"{77.4 + (time.time() % 100) / 1000.0:.6f}"
    try:
        # Submit a fresh complaint
        print("Submitting new unique complaint text note...")
        sub_data = {
            "live_latitude": test_lat,
            "live_longitude": test_lon,
            "live_location_timestamp": datetime.utcnow().isoformat() + "Z",
            "text_note": unique_text
        }
        sub_res = requests.post(f"{BASE_URL}/submit-complaint", data=sub_data)
        if sub_res.status_code != 200:
            log_test_result("Unique Complaint Submission", False, f"Status code: {sub_res.status_code}, error: {sub_res.text}")
            return
        log_test_result("Unique Complaint Submission", True, "Successfully submitted new complaint")

        # Wait a moment for DB commit
        time.sleep(1)

        # Retrieve updated dashboard
        dashboard_res = requests.get(f"{BASE_URL}/analytics/dashboard", headers=headers)
        updated_data = dashboard_res.json()
        up_stats = updated_data.get("complaint_stats", {})
        
        total_diff = up_stats.get("total_complaints_processed", 0) - prev_total
        unique_diff = up_stats.get("unique_complaints", 0) - prev_unique
        
        log_test_result("Processed count incremented", total_diff == 1, f"Previous: {prev_total}, New: {up_stats.get('total_complaints_processed')}")
        log_test_result("Unique count incremented", unique_diff == 1, f"Previous: {prev_unique}, New: {up_stats.get('unique_complaints')}")

        # Submit duplicate
        print("Submitting the exact same complaint to test duplicate categorization routing analytics...")
        dup_res = requests.post(f"{BASE_URL}/submit-complaint", data=sub_data)
        if dup_res.status_code != 200:
            log_test_result("Duplicate Complaint Submission", False, f"Status code: {dup_res.status_code}, error: {dup_res.text}")
            return
        
        dup_json = dup_res.json()
        is_dup_msg = "This issue already exists" in dup_json.get("message", "") or "Duplicate" in dup_json.get("message", "")
        log_test_result("Backend duplicate warning response", is_dup_msg, f"Message: {dup_json.get('message')}")

        # Wait a moment for DB commit
        time.sleep(1)

        # Retrieve updated dashboard again
        dashboard_res = requests.get(f"{BASE_URL}/analytics/dashboard", headers=headers)
        updated_data2 = dashboard_res.json()
        up_stats2 = updated_data2.get("complaint_stats", {})
        
        total_diff2 = up_stats2.get("total_complaints_processed", 0) - prev_total
        dup_diff2 = up_stats2.get("duplicate_complaints", 0) - prev_dup
        
        log_test_result("Processed count incremented for duplicate", total_diff2 == 2, f"Total Processed: {up_stats2.get('total_complaints_processed')}")
        log_test_result("Duplicate count incremented", dup_diff2 == 1, f"Duplicates Detected: {up_stats2.get('duplicate_complaints')}")

        # Energy saved verify
        energy_saved_val = updated_data2.get("energy_stats", {}).get("energy_saved_by_dedup", 0.0)
        log_test_result("Energy Saved by Deduplication tracked", energy_saved_val > 0.0, f"Energy Saved: {energy_saved_val} J")

    except Exception as e:
        log_test_result("E2E Pipeline Integration", False, f"Exception: {e}")

    print("\n" + "=" * 60)
    print("ALL DATA ANALYTES TESTING COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
