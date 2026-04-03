import requests
import json

try:
    with open("test.wav", "rb") as f:
        print("Submitting test.wav...")
        files = {"file": ("test.wav", f, "audio/wav")}
        data = {
            "live_latitude": "12.971599",
            "live_longitude": "77.594566",
            "live_location_timestamp": "2026-04-03T12:00:00Z",
            "text_note": "Pothole near signal"
        }
        response = requests.post("http://localhost:8000/submit-complaint", files=files, data=data)
        print("Status Code:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
