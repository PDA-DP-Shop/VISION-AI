import requests
import os
import json

def test_prediction(file_path):
    url = "http://127.0.0.1:5001/predict"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'image/jpeg')}
        try:
            r = requests.post(url, files=files)
            if r.status_code == 200:
                result = r.json()
                print(f"\n--- Results for {os.path.basename(file_path)} ---")
                print(f"Prediction: {result.get('prediction')}")
                print(f"Status: {result.get('status')}")
                print(f"Confidence: {result.get('confidence')}%")
                print(f"Reason: {result.get('reason')}")
                print(f"Detailed Scores: {result.get('detailed_scores')}")
            else:
                print(f"Error {r.status_code} for {file_path}: {r.text}")
        except Exception as e:
            print(f"Error testing {file_path}: {e}")

if __name__ == "__main__":
    # Test with existing samples
    samples = [
        "testing/sample_aadhaar.jpg",
        "testing/sample_pan.jpg"
    ]
    
    for sample in samples:
        test_prediction(sample)
