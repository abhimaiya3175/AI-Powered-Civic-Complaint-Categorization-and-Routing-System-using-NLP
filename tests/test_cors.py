import requests

url = "http://localhost:8000/complaints"
headers = {
    "Origin": "http://localhost:5173",
    "Access-Control-Request-Method": "GET",
    "Access-Control-Request-Headers": "authorization"
}
try:
    r = requests.options(url, headers=headers)
    print("Status Code:", r.status_code)
    print("Headers:", r.headers)
except Exception as e:
    print(e)
