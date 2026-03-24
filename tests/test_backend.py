import requests

base_url = "http://localhost:8000"

try:
    print("Logging in...")
    login = requests.post(f"{base_url}/login", data={"username": "admin", "password": "bbmp2025"})
    print("Login:", login.status_code)
    
    if login.status_code == 200:
        token = login.json()["access_token"]
        print("Got token.")
        
        print("Fetching stats...")
        stats = requests.get(f"{base_url}/complaints/stats", headers={"Authorization": f"Bearer {token}"})
        print("Stats Status:", stats.status_code)
        if stats.status_code != 200:
            print("Stats Error:", stats.text)
        else:
            print("Stats Data:", stats.json()[:100])
        
        print("Fetching complaints...")
        comps = requests.get(f"{base_url}/complaints?page=1&size=10", headers={"Authorization": f"Bearer {token}"})
        print("Complaints Status:", comps.status_code)
        if comps.status_code != 200:
            print("Complaints Error:", comps.text)
        else:
            print("Complaints Data:", str(comps.json())[:100])
except Exception as e:
    print(e)
