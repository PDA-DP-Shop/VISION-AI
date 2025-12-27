import os
import cv2
import numpy as np
import uuid
from PIL import Image, ImageChops, ImageEnhance

# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.preprocessing import image as keras_image
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.applications import EfficientNetB0

# --- COPIED UTILS FROM APP.PY TO AVOID FLASK CONTEXT ISSUES ---

MODEL_PATH = 'sentry_forensic_v3.h5'

def build_model():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# if os.path.exists(MODEL_PATH):
#     model = load_model(MODEL_PATH)
# else:
#     model = build_model()

def prepare_image(filepath, target_size=(224, 224)):
    img = keras_image.load_img(filepath, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Face Cascade
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(filepath):
    try:
        img = cv2.imread(filepath)
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0: return img 
        max_area = 0
        best_face = img
        for (x,y,w,h) in faces:
            if w*h > max_area:
                max_area = w*h
                best_face = img[y:y+h, x:x+w]
        return best_face
    except: return None

def analyze_fft(image_data):
    try:
        if image_data is None: return 0.0
        if len(image_data.shape) == 3:
            img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        else:
            img = image_data
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        mask_size = 30
        magnitude_spectrum[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
        mean_mag = np.mean(magnitude_spectrum)
        std_mag = np.std(magnitude_spectrum)
        return min(100.0, (std_mag / mean_mag) * 50.0)
    except: return 0.0

def generate_ela_score(filepath):
    try:
        original = Image.open(filepath).convert('RGB')
        temp_ela = f'temp_{uuid.uuid4().hex}.jpg'
        original.save(temp_ela, 'JPEG', quality=75)
        resaved = Image.open(temp_ela)
        ela_image = ImageChops.difference(original, resaved)
        ela_array = np.array(ela_image)
        ela_score = np.mean(ela_array)
        scaled = min(100.0, (ela_score / 10.0) * 100.0)
        if os.path.exists(temp_ela): os.remove(temp_ela)
        return scaled
    except: return 0.0

# --- USER IMAGES ---
images = [
    "/Users/devanshpatel/.gemini/antigravity/brain/cb70233f-f68c-4d29-b308-6d2c2b89fe32/uploaded_image_0_1766325565950.jpg",
    "/Users/devanshpatel/.gemini/antigravity/brain/cb70233f-f68c-4d29-b308-6d2c2b89fe32/uploaded_image_1_1766325565950.jpg",
    "/Users/devanshpatel/.gemini/antigravity/brain/cb70233f-f68c-4d29-b308-6d2c2b89fe32/uploaded_image_2_1766325565950.jpg",
    "/Users/devanshpatel/.gemini/antigravity/brain/cb70233f-f68c-4d29-b308-6d2c2b89fe32/uploaded_image_3_1766325565950.jpg",
    "/Users/devanshpatel/.gemini/antigravity/brain/cb70233f-f68c-4d29-b308-6d2c2b89fe32/uploaded_image_4_1766325565950.jpg"
]

print("-" * 50)
print("DEBUG ANALYSIS OF REAL IMAGES")
print("-" * 50)

for img_path in images:
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue
    
    # 1. Measurements
    fft = analyze_fft(cv2.imread(img_path))
    ela = generate_ela_score(img_path)
    
    face = extract_face(img_path)
    face_fft = analyze_fft(face)
    
    # prep_img = prepare_image(img_path)
    # model_raw = model.predict(prep_img, verbose=0)[0][0]
    ai_prob_model = 0.5 # Assume neutral model for now since we can't run TF

    # 2. Logic Simulation (Current app.py logic)
    combined_ai_score = (ai_prob_model * 0.30) + ((fft / 100.0) * 0.40) + ((ela / 100.0) * 0.30)
    
    reason = "Normal"
    verdict = "REAL"
    
    # Override Logic
    if ai_prob_model > 0.90:
        combined_ai_score = max(combined_ai_score, ai_prob_model)
        reason = "Veto: Model High Conf"
        
    if combined_ai_score < 0.80:
        if fft < 12: 
            combined_ai_score = 0.92
            reason = "Veto: Low Global FFT (Digital Purity)"
        elif face_fft < 25:
             combined_ai_score = 0.94
             reason = f"Veto: Low Face FFT ({face_fft:.1f}) (Smooth Face)"

    if combined_ai_score > 0.50:
        verdict = "FAKE"

    print(f"\nImage: {os.path.basename(img_path)}")
    print(f"  > FFT (Global): {fft:.2f}")
    print(f"  > FFT (Face):   {face_fft:.2f}")
    print(f"  > ELA Score:    {ela:.2f}")
    print(f"  > Model (AI%):  {ai_prob_model:.2f}")
    print(f"  > Combined:     {combined_ai_score:.2f}")
    print(f"  > Verdict:      {verdict} ({reason})")
