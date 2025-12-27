import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

print(f"API Key found: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:5001",
    "X-Title": "DeepfakeDetector"
}

# Try a model that was previously known to be free or cheap
model = "google/gemini-2.0-flash-exp:free"

data = {
    "model": model,
    "messages": [
        {"role": "user", "content": "Test"}
    ]
}

print(f"Testing model: {model}")
try:
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
