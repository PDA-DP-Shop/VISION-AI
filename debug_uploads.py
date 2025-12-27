import requests
import os
import json

UPLOAD_DIR = "/Users/devanshpatel/Devansh/project/Deepfake AI/uploads"
API_URL = "http://127.0.0.1:5001/predict-doc"

files = [
    "AWS Training & Certification.png",
    "Saylor Academy awards.png",
    "Skill India.png",
    "Screenshot 2025-12-20 at 7.17.49 PM.png"
]

for filename in files:
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
        
    print(f"\n>>> Analyzing: {filename}")
    with open(file_path, 'rb') as f:
        r = requests.post(API_URL, files={'file': f})
        
    if r.status_code == 200:
        res = r.json()
        print(f"Prediction: {res.get('prediction')}")
        print(f"Status: {res.get('status')}")
        print(f"Confidence: {res.get('confidence')}%")
        print(f"Reason: {res.get('reason')}")
        print(f"Detailed Scores: {json.dumps(res.get('detailed_scores'), indent=2)}")
    else:
        print(f"Error: {r.status_code} - {r.text}")
