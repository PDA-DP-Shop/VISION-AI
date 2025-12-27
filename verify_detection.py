
import os
import cv2
import numpy as np
from app import analyze_fft, generate_ela
import uuid
import glob

# Ensure static folder exists for ELA
os.makedirs('static', exist_ok=True)

def test_images():
    print("--- Verifying Detection Logic ---")
    
    # Grab some images from static to test
    images = glob.glob("static/*.png") + glob.glob("static/*.jpg")
    if not images:
        print("No images found in static/ to test. Please add some.")
        return

    print(f"Testing {len(images)} images...")
    
    for img_path in images[:5]: # Test first 5
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        # Test Face Extraction
        from app import extract_face
        try:
            face = extract_face(img_path)
            h, w = face.shape[:2]
            print(f"  Face Extraction: {w}x{h} (Original used if full size)")
        except Exception as e:
            print(f"  Face Extract Failed: {e}")
            face = cv2.imread(img_path)

        # Test FFT on Face
        try:
            fft_score = analyze_fft(face)
            print(f"  FFT Score (Face): {fft_score:.2f}")
        except Exception as e:
            print(f"  FFT Failed: {e}")

        # Test ELA
        try:
            ela_path, ela_score = generate_ela(img_path)
            print(f"  ELA Score: {ela_score:.2f}")
        except Exception as e:
            print(f"  ELA Failed: {e}")

if __name__ == "__main__":
    test_images()
