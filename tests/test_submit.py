import requests
import json

try:
    with open("test.wav", "rb") as f:
        print("Submitting test.wav...")
        files = {"file": ("test.wav", f, "audio/wav")}
        response = requests.post("http://localhost:8000/submit-complaint", files=files)
        print("Status Code:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
