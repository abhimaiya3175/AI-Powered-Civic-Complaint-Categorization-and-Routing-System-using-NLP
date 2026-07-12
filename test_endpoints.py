import json
from fastapi.testclient import TestClient
from main import app, get_current_user

app.dependency_overrides[get_current_user] = lambda: {"username": "testadmin"}
client = TestClient(app)

print("\n--- Testing Admin Panel Complaints Mismatch Filter ---")
response = client.get("/complaints?category_mismatch=true")
print("Status Code:", response.status_code)
try:
    data = response.json()
    print(f"Returned {len(data)} mismatched complaints.")
    if len(data) > 0:
        print("Sample:", json.dumps(data[0], indent=2))
except Exception:
    print(response.text)

print("\n--- Testing Analytics Dashboard ---")
response = client.get("/analytics/dashboard")
print("Status Code:", response.status_code)
try:
    data = response.json()
    print("Severity Distribution:", json.dumps(data.get("severity_distribution", {}), indent=2))
    print("Image Analysis Timing:", json.dumps(data.get("timing_metrics", {}).get("image_analysis", {}), indent=2))
except Exception:
    print(response.text)
