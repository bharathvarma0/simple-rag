
import requests
import json

url = "http://localhost:8000/api/v1/query"
payload = {"question": "What are the 2026 F1 engine regulations?"}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    print("Status Code:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, 'response') and e.response:
        print("Server response:", e.response.text)
