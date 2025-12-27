import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2

MODEL_PATH = 'sentry_forensic_v3.h5'

def create_random_image():
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

def create_solid_image(color):
    return np.full((224, 224, 3), color, dtype=np.uint8)

def verify():
    print(f"Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        print("Model file not found!")
        return

    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Try building it (in case it was saved weights-only or architecture mismatch)
        return

    # Test cases: Random Noise, White, Black
    # If the model is untrained/random, scores will likely be around 0.5 or very inconsistent for structured vs unstructured data.
    
    test_images = {
        "Random Noise": create_random_image(),
        "Solid White": create_solid_image(255),
        "Solid Black": create_solid_image(0)
    }

    print("\n--- Predictions (0=Fake, 1=Real) ---")
    
    for name, img in test_images.items():
        # Preprocess
        img_array = np.expand_dims(img, axis=0)
        prepped = preprocess_input(img_array)
        
        try:
            prediction = model.predict(prepped, verbose=0)[0][0]
            print(f"{name}: {prediction:.4f}")
        except Exception as e:
            print(f"{name}: Error - {e}")

    # Check weights for randomness? 
    # Hard to tell definitively without stats, but if 'Random Noise' gives high confidence (0.0 or 1.0), it's weird. 
    # If it gives ~0.5, it's uncertain.
    
    # Let's check the weights of the top dense layer
    try:
        last_layer = model.layers[-1]
        weights = last_layer.get_weights()
        if weights:
            w, b = weights
            print(f"\nLast Layer Weights Stats: Mean={np.mean(w):.4f}, Std={np.std(w):.4f}")
            if np.std(w) < 0.01:
                print("WARNING: Weights seem very uniform/small. Possibly untrained.")
    except:
        pass

if __name__ == "__main__":
    verify()
