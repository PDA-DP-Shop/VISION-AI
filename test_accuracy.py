import requests
import os

def test_prediction(file_path):
    url = "http://127.0.0.1:5001/predict-doc"
    with open(file_path, 'rb') as f:
        files = {'file': f}
        try:
            r = requests.post(url, files=files)
            print(f"Results for {os.path.basename(file_path)}:")
            print(r.json())
        except Exception as e:
            print(f"Error testing {file_path}: {e}")

if __name__ == "__main__":
    test_prediction("testing/sample_aadhaar.jpg")
    test_prediction("testing/sample_pan.jpg")
