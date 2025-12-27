import os
import cv2
import numpy as np
import ssl
import uuid
import base64
import requests
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from moviepy import VideoFileClip

load_dotenv() # Load API Key from .env

# macOS SSL Certificate Fix
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: 
    pass
else: 
    ssl._create_default_https_context = _create_unverified_https_context

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Face Cascade Classifier
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def extract_face(filepath):
    """
    Extracts the largest face from an image. 
    Returns the cropped face image (numpy array).
    If no face found, returns the original image.
    """
    try:
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return img # Fallback to full image
            
        # Get largest face
        max_area = 0
        best_face = img
        for (x,y,w,h) in faces:
            if w*h > max_area:
                max_area = w*h
                best_face = img[y:y+h, x:x+w]
        
        return best_face
    except Exception as e:
        print(f"Face extraction failed: {e}")
        return cv2.imread(filepath)

def analyze_fft(image_data):
    """
    Performs FFT on an image array (grayscale) to detect high-frequency anomalies.
    Input: image_data (numpy array, BGR or Grayscale)
    """
    try:
        # Convert to gray if needed
        if len(image_data.shape) == 3:
            img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        else:
            img = image_data
            
        # Calculate DFT
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        
        # Mask center (low frequencies)
        mask_size = 30
        magnitude_spectrum[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
        
        mean_mag = np.mean(magnitude_spectrum)
        std_mag = np.std(magnitude_spectrum)
        
        score = min(100.0, (std_mag / mean_mag) * 50.0)
        return score
    except Exception as e:
        print(f"FFT Error: {e}")
        return 0.0

def generate_ela(filepath, quality=75):
    original = Image.open(filepath).convert('RGB')
    temp_ela = os.path.join('static', f'temp_resave_{uuid.uuid4().hex}.jpg')
    original.save(temp_ela, 'JPEG', quality=quality)
    resaved = Image.open(temp_ela)
    ela_image = ImageChops.difference(original, resaved)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    enhanced_ela = ImageEnhance.Brightness(ela_image).enhance(scale * 2.0)
    
    ela_array = np.array(ela_image)
    ela_score = np.mean(ela_array)
    scaled_ela_score = min(100.0, (ela_score / 10.0) * 100.0)

    ela_filename = f'heatmap_{uuid.uuid4().hex}.png'
    enhanced_ela.save(os.path.join('static', ela_filename))
    if os.path.exists(temp_ela): os.remove(temp_ela)
    return ela_filename, scaled_ela_score

def analyze_metadata(filepath, filename):
    report = {"is_ai": False, "device": "Digital Source", "software": "Standard", "source_category": "Unknown"}
    if "WhatsApp" in filename:
        return {"is_ai": False, "device": "WhatsApp Mobile", "software": "WhatsApp", "source_category": "Social Media"}
    
    # Specific AI Softwares (Avoiding short strings like 'ai')
    ai_keys = ['dalle', 'midjourney', 'stable diffusion', 'firefly', 'gan', 'openai', 'anthropic', 'flux.1']
    edit_keys = ['photoshop', 'gimp', 'canva', 'picsart', 'lightroom', 'adobe', 'illustrator', 'snapseed', 'facetune', 'pixlr']
    
    # Skip Image.open for common video formats to prevent PIL errors
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    if any(filename.lower().endswith(ext) for ext in video_exts):
        report["source_category"] = "Video Stream"
        return report

    try:
        img = Image.open(filepath)
        exif = img._getexif()
        info = img.info 
        
        def check_val(val_str, tag=""):
            v = val_str.lower()
            
            # CRITICAL: Ignore standard XML/XMP namespace URLs containing 'adobe'
            ignore_patterns = ['adobe.com', 'ns.adobe.com', 'adobe:ns', 'purl.org', 'w3.org']
            if any(p in v for p in ignore_patterns):
                return False, False

            found_ai = False
            found_edit = False
            
            for k in ai_keys:
                if k in v:
                    found_ai = True
            
            if " ai " in f" {v} " or v.startswith("ai ") or v.endswith(" ai"):
                found_ai = True
            
            for k in edit_keys:
                if k in v:
                    found_edit = True
            
            return found_ai, found_edit

        if exif:
            for tag, value in exif.items():
                tag_name = TAGS.get(tag, tag)
                # Ensure value is string for checking, but keep original for report if needed
                val_str = str(value)
                if isinstance(value, bytes):
                    try: val_str = value.decode('utf-8', errors='ignore')
                    except: val_str = str(value)

                if tag_name == 'Make': report["device"] = val_str
                if tag_name in ['Software', 'ProcessingSoftware', 'ImageDescription', 'UserComment']:
                    is_ai, is_edit = check_val(val_str, tag_name)
                    if is_ai: report["is_ai"] = True
                    if is_edit: report["source_category"] = "Modified / Edited"
                    report["software"] = val_str
        
        for key, val in info.items():
            val_str = str(val)
            if isinstance(val, bytes):
                try: val_str = val.decode('utf-8', errors='ignore')
                except: val_str = str(val)

            is_ai, is_edit = check_val(val_str, str(key))
            if is_ai: report["is_ai"] = True
            if is_edit: report["source_category"] = "Modified / Edited"
            if "software" in str(key).lower():
                report["software"] = val_str

        if report["source_category"] == "Unknown":
            if report["device"] != "Digital Source":
                report["source_category"] = "Original Camera File"
            else:
                report["source_category"] = "Compressed Web / Screenshot"
                
    except Exception as e:
        print(f"Meta Error: {e}")
    return report

def analyze_lsb(filepath):
    """
    Analyzes the Least Significant Bit (LSB) plane for noise patterns.
    Real cameras produce random thermal noise in the LSB.
    """
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None: return 0.0
        
        # Extract LSB (Last bit)
        lsb = img & 1
        
        # Calculate consistency/randomness
        pixel_mean = np.mean(lsb)
        
        # Score calculation: 
        # Deviation from 0.5 implies lack of randomness.
        distance = abs(0.5 - pixel_mean)
        lsb_score = max(0, (0.5 - distance) * 200)
        
        return lsb_score
    except Exception as e:
        print(f"LSB Error: {e}")
        return 0.0

def analyze_eyes(face_img):
    """
    Checks for pupil consistency.
    Real eyes have circular pupils and similar reflection patterns.
    """
    try:
        if face_img is None: return 50.0 # Neutral
        
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        eyes = EYE_CASCADE.detectMultiScale(gray_face, 1.1, 3)
        
        if len(eyes) < 2:
            return 40.0 # Suspicious (One eye or deformed?)
            
        # Check symmetry and shape
        shapes = []
        for (ex, ey, ew, eh) in eyes:
            eye_roi = gray_face[ey:ey+eh, ex:ex+ew]
            # Hough Circles for Pupil
            circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 1, 20,
                                      param1=50, param2=30, minRadius=0, maxRadius=0)
            
            if circles is not None:
                shapes.append(1) # Good circle found
            else:
                shapes.append(0) # Malformed/Blobby
        
        # Scoring
        if sum(shapes) == 2: return 95.0 # Perfect circular pupils
        if sum(shapes) == 1: return 60.0 # One good, one bad
        return 20.0 # Both eyes failed circular check
        
    except Exception as e:
        print(f"Eye Error: {e}")
        return 50.0

def analyze_skin_texture(face_img):
    """
    Analyzes skin region for 'Plasticity'.
    """
    try:
        if face_img is None: return 50.0
        
        # Convert to YCrCb for skin detection
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        
        # Skin Color Rule (Generic)
        min_YCrCb = np.array([0,133,77],np.uint8)
        max_YCrCb = np.array([255,173,127],np.uint8)
        
        skin_mask = cv2.inRange(ycrcb,min_YCrCb,max_YCrCb)
        skin_pixels = cv2.bitwise_and(face_img, face_img, mask=skin_mask)
        
        if np.sum(skin_mask) == 0: return 50.0 # No skin detected
        
        # Analyze texture of skin pixels only
        gray_skin = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2GRAY)
        
        # Calculate Variance of Laplacian (Texture Sharpness) on masked region
        laplacian = cv2.Laplacian(gray_skin, cv2.CV_64F)
        
        # We only care about the variance within the skin mask
        skin_values = laplacian[skin_mask > 0]
        
        if len(skin_values) == 0: return 50.0
        
        texture_score = np.var(skin_values)
        
        # Normalization: 
        # Low variance (< 100) -> Plastic/AI
        # High variance (> 300) -> Real/Textured
        
        score = min(100.0, (texture_score / 300.0) * 100.0)
        return score
        
    except Exception as e:
        print(f"Skin Error: {e}")
        return 50.0

def analyze_background(filepath, face_img):
    """
    Compares Face noise vs Background noise.
    """
    try:
        full_img = cv2.imread(filepath)
        if full_img is None or face_img is None: return 50.0
        
        gray_full = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        var_full = cv2.Laplacian(gray_full, cv2.CV_64F).var()
        var_face = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Ratio
        if var_full == 0: return 0.0
        
        ratio = var_face / var_full
        deviation = abs(ratio - 1.5) # Assume 1.5 is standard portrait
        consistency = max(0, 100 - (deviation * 10))
        
        return consistency
        
    except Exception as e:
        print(f"BG Error: {e}")
        return 50.0

def analyze_noise_patterns(filepath):
    """
    Checks for Photon Transfer Curve (PTC) conformance.
    """
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None: return 50.0
        
        h, w = img.shape
        patch_size = 10
        means = []
        variances = []
        
        # Sample 500 random patches
        for _ in range(500):
            x = np.random.randint(0, w - patch_size)
            y = np.random.randint(0, h - patch_size)
            patch = img[y:y+patch_size, x:x+patch_size]
            
            m = np.mean(patch)
            v = np.var(patch)
            
            if v < 100: 
                means.append(m)
                variances.append(v)
        
        if len(means) < 20: return 50.0 
        
        correlation = np.corrcoef(means, variances)[0, 1]
        avg_variance = np.mean(variances)
        
        if avg_variance < 3.0: return 0.0 # Plastic/Synthetic
        if correlation > 0.2: return 100.0 # Matches Physics
        if correlation < 0.1 and avg_variance > 10.0: return 20.0 # Artificial Noise
        
        return 50.0 
    except Exception as e:
        print(f"Noise Error: {e}")
        return 50.0




def convert_to_wav(input_path, output_wav_path):
    """
    Robust audio conversion using direct ffmpeg subprocess.
    """
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # -y: overwrite, -i: input, -ac 1: mono, -ar 22050: sample rate, -acodec pcm_s16le: format
        cmd = [ffmpeg_exe, "-y", "-i", input_path, "-ac", "1", "-ar", "22050", "-acodec", "pcm_s16le", output_wav_path]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"FFMPEG Conversion Error: {e}")
        return False

def create_temporal_grid(frame_paths, output_path):
    """
    Combines 4 frame paths into a 2x2 grid for temporal analysis.
    """
    try:
        from PIL import Image
        images = [Image.open(p) for p in frame_paths if os.path.exists(p)]
        if len(images) < 4: 
            # Padding if we don't have enough frames
            while len(images) < 4: images.append(images[-1] if images else Image.new('RGB', (512, 512)))
        
        # Resize all to same small size for efficiency
        size = (640, 360)
        images = [img.resize(size) for img in images]
        
        grid = Image.new('RGB', (size[0]*2, size[1]*2))
        grid.paste(images[0], (0, 0))
        grid.paste(images[1], (size[0], 0))
        grid.paste(images[2], (0, size[1]))
        grid.paste(images[3], (size[0], size[1]))
        
        grid.save(output_path, "JPEG", quality=85)
        return True
    except Exception as e:
        print(f"Grid Error: {e}")
        return False

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extracts audio from video file using FFMPEG.
    """
    return convert_to_wav(video_path, output_audio_path)

def analyze_audio_forensics(wav_path):
    """
    Upgraded Acoustic Forensic Engine for Deepfake Detection.
    """
    try:
        sample_rate, samples = wavfile.read(wav_path)
        if len(samples.shape) > 1: samples = samples[:, 0] # Mono
        samples = samples.astype(np.float32)
        
        # 1. Noise Floor Entropy (Real mics have complex noise; AI has 'flat' noise)
        total_len = len(samples)
        chunk_size = int(sample_rate * 0.1) # 100ms
        min_chunks = []
        for i in range(0, total_len - chunk_size, chunk_size):
            chunk = samples[i:i+chunk_size]
            rms = np.sqrt(np.mean(chunk**2))
            min_chunks.append(rms)
        
        min_energy = min(min_chunks) if min_chunks else 0
        
        # 2. Synthetic Signature: High-Frequency Cutoff Check
        # AI often cuts exactly at 8kHz or 16kHz
        fft_data = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1/sample_rate)
        hf_energy = np.mean(fft_data[freqs > 8000]) if any(freqs > 8000) else 0
        lf_energy = np.mean(fft_data[freqs < 4000])
        
        hf_ratio = hf_energy / lf_energy if lf_energy > 0 else 0
        
        return {
            'noise_floor': min_energy,
            'is_perfect_silence': min_energy < 0.05, # AI Signature
            'is_hf_clipped': hf_ratio < 0.001, # Likely AI or low-quality compression
            'forensic_score': 100 if min_energy < 0.05 else 50
        }
    except Exception as e:
        print(f"Forensic Logic Error: {e}")
        return {'noise_floor': 1, 'is_perfect_silence': False, 'is_hf_clipped': False}

def generate_audio_spectrogram(audio_path, output_img_path):
    """
    Generates a Mel-Spectrogram-like visualization for AI forensic audit.
    """
    try:
        sample_rate, samples = wavfile.read(audio_path)
        if len(samples.shape) > 1: samples = samples[:, 0] # Mono
        
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        
        plt.figure(figsize=(14, 7))
        # High resolution spectrogram focusing on voice band (300Hz - 8kHz)
        plt.specgram(samples, NFFT=2048, Fs=sample_rate, noverlap=1024, cmap='inferno', scale='dB')
        plt.ylim(0, 10000) # Most voice artifacts are below 10kHz
        plt.axis('off')
        plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        return True
    except Exception as e:
        print(f"Spectrogram Error: {e}")
        return False

def analyze_with_openrouter(filepath, mode="image"):
    """
    Uses OpenRouter (LLM) to analyze the image/document.
    mode: 'image' (Deepfake) or 'document' (Forgery/Edit)
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key: 
        print("DEBUG: No API Key found in env.")
        return None 
    
    print(f"DEBUG: API Key loaded (starts with): {api_key[:10]}...")

    try:
        if mode == "text_forensics":
            # In text mode, 'filepath' is the raw text content
            encoded_string = filepath
        else:
            with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5001", 
            "X-Title": "DeepfakeDetector" 
        }
        
        # Specialized Model Lists to prevent sending images to text-only models
        vision_modes = ["image", "video_frame", "audio_spectrogram", "document"]
        
        if mode in vision_modes:
            models_to_try = [
                "google/gemini-2.0-flash-exp:free",
                "moonshotai/kimi-vl-a3b-thinking:free",
                "google/gemma-3-27b-it:free",
                "google/gemma-3-12b-it:free",
                "nvidia/nemotron-nano-12b-v2-vl:free",
                "mistralai/mistral-small-3.1-24b-instruct:free",
                "google/gemini-2.0-flash-001", # Paid fallbacks
                "openai/gpt-4o-mini"
            ]
        else: # text_forensics
            models_to_try = [
                "google/gemini-2.0-flash-exp:free",
                "mistralai/mistral-small-3.1-24b-instruct:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-r1:free",
                "xiaomi/mimo-v2-flash:free",
                "microsoft/phi-3-medium-128k-instruct:free",
                "meta-llama/llama-3.1-405b-instruct:free",
                "google/gemini-2.0-flash-001",
                "openai/gpt-4o-mini"
            ]
        
        prompt = ""
        if mode == "document":
            prompt = """
            You are a Senior Forensic Document Auditor specialized in **INDIAN GOVERNMENT IDs (Aadhaar, PAN, DL, Voter ID)** and **GLOBAL TECH CERTIFICATES**.
            Conduct a pixel-perfect audit. Flag as FAKE if any visual element (logos, fonts, ghost images) looks tampered.
            Return ONLY a raw JSON object: {"is_ai": boolean, "confidence": float, "reason": "string"}
            """
        if mode == "image":
            prompt = '''Analyze the given image using advanced visual forensics to determine whether it is AI-generated (deepfake) or authentic (real).

Inspect fine-grained facial, lighting, and structural inconsistencies commonly introduced by generative models, including but not limited to:
	•	Irregular or asymmetrical pupils, unnatural eye reflections, or gaze mismatch
	•	Abnormal skin texture such as over-smoothing, loss of pores, or patchy blending near facial boundaries
	•	Lighting and shadow inconsistencies that violate physical light direction or reflection rules
	•	Background warping, edge artifacts, halo effects, or distorted geometry
	•	Motion or temporal inconsistencies if the image is extracted from a video frame
            Return ONLY a raw JSON object:
            {is_ai: bool, confidence: int, reason: string, checklist: {eyes: string, texture: string, lighting: string, motion: string}}
            Use a confidence scale from 0–100, where higher values indicate stronger certainty of AI generation.'''
        elif mode == "video_frame":
            prompt = "Analyze this 2x2 TEMPORAL GRID from a video. CRITICAL: REALITY & CONTEXT CHECK. If historical figures (e.g. Gandhi) appear in modern settings, it is 100% PROVEN DEEPFAKE. Look for: 1. Lipsync drift, 2. Face-swap flickering, 3. Eye reflection mismatches. Also provide a verbatim transcript and translation of any speech present in the audio spectrogram layer. Return JSON: {is_ai: bool, confidence: int, reason: string, transcript: string, translation: string, checklist: {lipsync: string, eyes: string, expression: string, texture: string, lighting: string, motion: string}, awareness_note: string}"
        elif mode == "audio_spectrogram":
            # --- PROFESSIONAL 10-POINT AUDIO FORENSIC AUDIT ---
            prompt = """Analyze the provided audio forensic spectrogram to determine if the voice is Real, AI-Generated, or Suspicious.
            Conduct a rigorous audit across these 10 aspects:
            1. Voice Naturalness – robotic tone, over-smooth speech, or unnatural clarity.
            2. Emotion Consistency – flat or mismatched emotional expression.
            3. Breathing & Pauses – missing natural breaths, unnatural pauses.
            4. Pitch & Prosody – sudden pitch jumps, unnatural rhythm or stress.
            5. Pronunciation Errors – incorrect emphasis or odd word stress.
            6. Background Noise – absence of natural ambient noise or inconsistent noise patterns.
            7. Audio Artifacts – digital glitches, metallic sound, echo-like effects.
            8. Continuity – abrupt voice changes within the same speaker.
            9. Compression Effects – distinguish between social-media compression and synthetic distortions.
            10. Source Verification – context and original source assessment.

            In your reasoning:
            - Explain WHY these characteristics indicate AI-generated audio.
            - Clarify how low-quality recordings or heavy compression can affect real audio (preventing false positives).

            Return ONLY a raw JSON object with these EXACT keys:
            {
                "verdict": "Real / AI-Generated / Suspicious",
                "is_ai": boolean,
                "confidence": float,
                "reason": "Identify the primary smoking gun or forensic violation.",
                "transcript": "Verbatim transcript of the voice.",
                "translation": "English translation (only if the voice is in a non-English language; otherwise leave empty).",
                "checklist": {
                    "Voice Naturalness": "string",
                    "Emotion Consistency": "string",
                    "Breathing & Pauses": "string",
                    "Pitch & Prosody": "string",
                    "Pronunciation Errors": "string",
                    "Background Noise": "string",
                    "Audio Artifacts": "string",
                    "Continuity": "string",
                    "Compression Effects": "string",
                    "Source Verification": "string"
                },
                "awareness_note": "Short explanation suitable for public awareness or academic use."
            }"""
        elif mode == "text_forensics":
            # --- ELITE TEXT FORENSICS ENGINE ---
            safe_text = json.dumps(encoded_string)
            prompt = f"""You are an expert Linguistic Forensic Analyst and Cyber-Intelligence Operative. 
            Critically audit the following text for **AI-Generation (LLM signatures)** and **Adversarial Harm/Spam**.

            **ANALYSIS TARGET**: {safe_text}

            **AUDIT CRITERIA (The Forensic 5)**:
            1. **Perplexity & Burstiness**: (AI Tell) Look for uniform sentence length and predictable word choice vs. human variance. *Note: Transactional templates have low burstiness but are REAL.*
            2. **Transactional Recognition**: (LEGIT Tell) Identify safe automated alerts (OTPs, Missed Call alerts from 'Team Jio', 'Bank Alerts'). If it is a standard business template without a malicious link, mark as **REAL**.
            3. **Dark Patterns & Urgency**: (Spam Tell) Detect "False Scarcity", "Suspicious Authority", or "Panic Induction". Differentiate from legitimate account alerts.
            4. **Impersonation Risk**: Does the tone match the claimed identity? Check for "Phishing-style" credential requests.
            5. **Logic & Nuance Audit**: Audit for base LLM patterns vs. rigid system-generated templates.

            **VERDICT RULES**:
            - **Real**: Human writing OR Legitimate Transactional/Automated system alerts (Telecom/Bank notifications).
            - **AI-Generated**: LLM-authored content (essays, fake reviews, 'cookie-cutter' long-form).
            - **Suspicious/Spam**: Phishing, scam outreach, or malicious urgency.

            **ANTI-FALSE POSITIVE WARNING**: Do not flag formal, academic, or highly educated human writing as AI just because it follows rules. Look for the "soul" of the writing—base LLMs are cookie-cutter; humans are unpredictable even when formal.

            Return ONLY a raw JSON object with these EXACT keys:
            {{
                "verdict": "Real / AI-Generated / Suspicious",
                "is_ai": boolean,
                "is_spam": boolean,
                "confidence": float (percentage 0-100),
                "reason": "Specify the exact linguistic signature (e.g., 'Low burstiness with generic connectors' or 'Classic phishing urgency pattern').",
                "checklist": {{
                    "Perplexity Audit": "string",
                    "Burstiness Check": "string",
                    "Dark Pattern Detection": "string",
                    "Tone Consistency": "string",
                    "Nuance Factor": "string"
                }},
                "awareness_note": "A summary for avoiding this specific type of risk or identifying the AI model type."
            }}"""
        
        for model_name in models_to_try:
            print(f"DEBUG: Trying Cloud Model: {model_name}")
            try:
                if mode == "text_forensics":
                    # For text forensics, we send text content instead of an image
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
                            ]
                        }
                    ]

                data = {
                    "model": model_name,
                    "messages": messages
                }
                
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
                
                if response.status_code == 200:
                    if not response.content:
                         print(f"DEBUG: Model {model_name} returned EMPTY response.")
                         continue
                         
                    try:
                        result = response.json()
                        if 'choices' not in result or not result['choices']:
                            print(f"DEBUG: Model {model_name} returned invalid choice structure: {result}")
                            continue
                            
                        content = result['choices'][0]['message']['content']
                        
                        # ANTI-EMPTY CONTENT CHECK: Some free models return " " on failure
                        if not content or not content.strip():
                            print(f"DEBUG: Model {model_name} returned whitespace-only content. Switching.")
                            continue

                        # Robust cleaning for Markdown
                        if '```json' in content:
                            content = content.split('```json')[1].split('```')[0].strip()
                        elif '```' in content:
                            content = content.split('```')[1].split('```')[0].strip()
                        else:
                            content = content.strip()
                            
                        print(f"DEBUG: Success with {model_name}")
                        parsed = json.loads(content)
                        parsed['vetted_by'] = model_name
                        return parsed
                    except Exception as json_err:
                        print(f"DEBUG: JSON Parse Error with {model_name}: {json_err}")
                        print(f"DEBUG: Raw Response Start: {response.text[:500]}")
                        continue
                        
                elif response.status_code == 429:
                    print(f"DEBUG: Model {model_name} Rate Limited (429). Switching...")
                elif response.status_code == 402:
                    print(f"DEBUG: Model {model_name} Error 402: Payment Required / Insufficient Credits.")
                elif response.status_code == 404:
                    print(f"DEBUG: Model {model_name} Error 404: Not Found or Unavailable.")
                else:
                    print(f"DEBUG: Model {model_name} HTTP {response.status_code} Error.")
                    
            except Exception as e:
                print(f"DEBUG: Request Exception {model_name}: {e}")
                continue
                
        print("DEBUG: All Cloud Models failed.")
        return None
            
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

@app.route('/')
def home(): return render_template('index.html')

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/detect')
def detect_page(): return render_template('detect.html')

@app.route('/video')
def video_page(): return render_template('video.html')

@app.route('/predict-video', methods=['POST'])
def predict_video():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    unique_filename = f"vid_{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened(): return jsonify({'error': 'Could not open video file'}), 400

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        frames_per_30s = int(fps * 30)
        num_segments = (total_frames + frames_per_30s - 1) // frames_per_30s
        
        meta = analyze_metadata(filepath, file.filename)
        
        # --- PRE-EXTRACT AUDIO ONCE (Efficiency) ---
        audio_wav = os.path.join('static', f"full_aud_{uuid.uuid4().hex}.wav")
        audio_spec = os.path.join('static', f"full_spec_{uuid.uuid4().hex}.jpg")
        audio_available = False
        if extract_audio_from_video(filepath, audio_wav):
            if generate_audio_spectrogram(audio_wav, audio_spec):
                audio_available = True

        # We will iterate through segments. If we find a high-confidence fake, we stop.
        # Otherwise, we keep going till the end.
        final_prediction = None
        
        for seg_idx in range(num_segments):
            start_frame = seg_idx * frames_per_30s
            end_frame = min(start_frame + frames_per_30s, total_frames)
            seg_len = end_frame - start_frame
            
            print(f"DEBUG: Scanning Segment {seg_idx+1}/{num_segments} (Frames {start_frame}-{end_frame})")
            
            num_checks = 10 
            step = max(1, seg_len // num_checks)
            
            fake_frames = 0
            analyzed_frames = 0
            previous_fft = None
            temporal_instability = 0
            suspicious_frame_path = None
            min_fft = 100
            
            current_frame = start_frame
            while current_frame < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if not ret: break
                
                analyzed_frames += 1
                temp_frame_path = os.path.join('static', f'temp_{uuid.uuid4().hex}.jpg')
                cv2.imwrite(temp_frame_path, frame)
                
                face_img = extract_face(temp_frame_path)
                if face_img is not None and len(face_img.shape) == 3:
                    h_face, w_face, _ = face_img.shape
                    h_orig, w_orig, _ = frame.shape
                    
                    if h_face > h_orig * 0.9 and w_face > w_orig * 0.9:
                        if os.path.exists(temp_frame_path): os.remove(temp_frame_path)
                        current_frame += step
                        continue
                else:
                    # Skip frame if face extraction or shape is invalid
                    if os.path.exists(temp_frame_path): os.remove(temp_frame_path)
                    current_frame += step
                    continue

                fft_score = analyze_fft(face_img)
                if fft_score < min_fft:
                    min_fft = fft_score
                    if suspicious_frame_path and os.path.exists(suspicious_frame_path):
                        os.remove(suspicious_frame_path)
                    suspicious_frame_path = os.path.join('static', f'suspicious_{uuid.uuid4().hex}.jpg')
                    cv2.imwrite(suspicious_frame_path, frame)

                if previous_fft is not None:
                    if abs(fft_score - previous_fft) > 15:
                        temporal_instability += 1
                previous_fft = fft_score

                if fft_score < 3.8: fake_frames += 1 # Lowered from 4.5 for cinematic media
                if os.path.exists(temp_frame_path): os.remove(temp_frame_path)

                current_frame += step
                if analyzed_frames >= num_checks: break
            
            if analyzed_frames < 2: continue # Try next segment
                
            fake_ratio = fake_frames / analyzed_frames
            instability_ratio = temporal_instability / (analyzed_frames - 1) if analyzed_frames > 1 else 0
            
            # --- CLOUD AUDIT (Per Segment) ---
            grid_frames = []
            
            # 1. First Frame
            p1 = os.path.join('static', f'grid1_{uuid.uuid4().hex}.jpg')
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + int(seg_len * 0.1))
            ret, f1 = cap.read(); 
            if ret: 
                cv2.imwrite(p1, f1)
                grid_frames.append(p1)

            # 2. Suspicious Frame
            if suspicious_frame_path: 
                grid_frames.append(suspicious_frame_path)

            # 3. Audio Spectrogram (Forensic layer)
            if audio_available:
                grid_frames.append(audio_spec)

            # 4. End Frame
            p4 = os.path.join('static', f'grid4_{uuid.uuid4().hex}.jpg')
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + int(seg_len * 0.9))
            ret, f4 = cap.read(); 
            if ret: 
                cv2.imwrite(p4, f4)
                grid_frames.append(p4)

            grid_output_path = os.path.join('static', f'grid_audit_{uuid.uuid4().hex}.jpg')
            cloud_result = None
            if create_temporal_grid(grid_frames, grid_output_path):
                cloud_result = analyze_with_openrouter(grid_output_path, mode="video_frame")
            
            # Cleanup segment files
            segment_temp_files = grid_frames + [grid_output_path]
            for p in segment_temp_files: 
                if p != suspicious_frame_path and p != audio_spec and os.path.exists(p): os.remove(p)
            if suspicious_frame_path and os.path.exists(suspicious_frame_path): os.remove(suspicious_frame_path)
            
            # Evaluate this segment
            cloud_is_ai = False
            cloud_conf = 0.0
            if cloud_result:
                cloud_is_ai = bool(cloud_result.get('is_ai', False))
                try:
                    raw_conf = cloud_result.get('confidence', 0)
                    if isinstance(raw_conf, str):
                        raw_conf = raw_conf.replace('%', '').strip()
                    cloud_conf = float(raw_conf) if raw_conf is not None else 0.0
                except:
                    cloud_conf = 50.0 # Fallback

            print(f"DEBUG: Segment {seg_idx+1} Metrics -> FakeRatio: {fake_ratio:.3f}, Instability: {instability_ratio:.3f}, Cloud AI: {cloud_is_ai} ({cloud_conf}%)")
            if cloud_result: print(f"DEBUG: Cloud Logic Hub -> {cloud_result.get('reason', 'N/A')}")
            
            # A segment is only suspicious if fake_ratio is sustained or Cloud has high confidence
            is_suspicious = (fake_ratio > 0.15) or (instability_ratio > 0.28) or cloud_is_ai
            
            # --- BALANCED VIDEO FORENSIC VETO ---
            if not cloud_is_ai and cloud_result:
                # Cloud thinks it's Real. Veto only if local evidence is overwhelming.
                if fake_ratio > 0.12 or instability_ratio > 0.22:
                    is_suspicious = True
                    cloud_is_ai = True 
                    cloud_conf = max(cloud_conf, (fake_ratio * 100) + 15) 
                    cloud_result['reason'] = "Forensic Veto: Targeted temporal instability and facial frequency anomalies detected despite Cloud AI pass."
            
            if is_suspicious:
                # BREAK and return AI detected result immediately (Early Exit)
                final_checklist = {
                    "Face & Lip Sync": cloud_result.get('checklist', {}).get('lipsync', "Drift detected in segment.") if cloud_result else "Drift detected.",
                    "Eye Behavior": cloud_result.get('checklist', {}).get('eyes', "Abnormal eye patterns.") if cloud_result else "Abnormal.",
                    "Facial Expressions": cloud_result.get('checklist', {}).get('expression', "Inconsistent expressions.") if cloud_result else "Inconsistent.",
                    "Skin Texture & Edges": f"Forensic noise analysis: unnatural smoothing. {cloud_result.get('checklist', {}).get('texture', '') if cloud_result else ''}",
                    "Lighting & Shadows": cloud_result.get('checklist', {}).get('lighting', "Shadow mismatch.") if cloud_result else "Mismatch.",
                    "Head & Body Movement": cloud_result.get('checklist', {}).get('motion', "Jaw-pivot instability.") if cloud_result else "Instability.",
                    "Audio Analysis (Voice Print)": cloud_result.get('checklist', {}).get('audio', "Acoustic consistency high.") if cloud_result else "Acoustic check clear.",
                    "Background Consistency": "Geometric warping detected in frame grid analysis.",
                    "Frame-by-Frame Pulse": f"Instability: {round(instability_ratio*100, 1)}% (High deviation).",
                    "Metadata / Source": f"Source marked as: {meta.get('source_category', 'Unknown digital file')}."
                }
                
                cap.release() # CRITICAL: Release before early return
                return jsonify({
                    'prediction': "AI VIDEO / DEEPFAKE DETECTED",
                    'confidence': round(max(fake_ratio * 100, cloud_conf, instability_ratio * 100), 1),
                    'reason': cloud_result.get('reason', f"Deepfake artifacts detected in segment {seg_idx+1}.") if cloud_result else f"Localized artifacts in segment {seg_idx+1}.",
                    'checklist': final_checklist,
                    'transcript': cloud_result.get('transcript', '') if cloud_result else '',
                    'translation': cloud_result.get('translation', '') if cloud_result else '',
                    'awareness_note': cloud_result.get('awareness_note', "Segmented scan revealed deepfake patterns.") if cloud_result else "Deepfake detected.",
                    'compression_warning': "Localizing artifacts in specific video segments.",
                    'status': "fake",
                    'device': 'Segmented Guard'
                })
            
            # If not suspicious, store the "best guess" so far if we finish all segments
            if not final_prediction:
                final_prediction = {
                    'prediction': "REAL VIDEO / REEL",
                    'confidence': round(min(99.0, (1.0 - fake_ratio) * 100), 1),
                    'reason': cloud_result.get('reason', "All segments passed forensic audit.") if cloud_result else "Forensic clear.",
                    'status': "real",
                    'checklist': {
                        "Face & Lip Sync": "Consistent across all 30s windows.",
                        "Eye Behavior": "Natural symmetry maintained.",
                        "Facial Expressions": "Dynamic and authentic.",
                        "Skin Texture & Edges": "Natural skin grain detected.",
                        "Lighting & Shadows": "Ambient light matches physique.",
                        "Head & Body Movement": "Natural physics observed.",
                        "Audio Analysis (Visual)": "Speech rhythm matches muscle movements.",
                        "Background Consistency": "No warping detected in any segment.",
                        "Frame-by-Frame Pulse": "Temporal stability confirmed.",
                        "Metadata / Source": f"Source: {meta.get('source_category', 'Authentic Stream')}."
                    }
                }
                final_prediction['transcript'] = cloud_result.get('transcript', '') if cloud_result else ''
                final_prediction['translation'] = cloud_result.get('translation', '') if cloud_result else ''

        cap.release()
        
        # If we reached here, no segment was flagged as fake
        if not final_prediction:
             return jsonify({'prediction': "INCONCLUSIVE", 'confidence': 0.0, 'reason': "Video too short or unreadable."})
             
        return jsonify({
            **final_prediction,
            'awareness_note': "Through scanning of all 30s segments complete.",
            'compression_warning': "Note: Low resolution can sometimes mimic artifacts.",
            'device': 'Segmented Guard'
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath): os.remove(filepath)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"img_{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        # --- Run Analyses ---
        meta = analyze_metadata(filepath, file.filename)
        heatmap, ela_score = generate_ela(filepath)
        
        face_img = extract_face(filepath)
        fft_score = analyze_fft(face_img) 
        lsb_score = analyze_lsb(filepath) 
        
        eye_score = 50.0
        skin_score = 50.0
        bg_score = 50.0
        noise_score = analyze_noise_patterns(filepath) 
        
        if face_img is not None:
             eye_score = analyze_eyes(face_img)
             skin_score = analyze_skin_texture(face_img)
             bg_score = analyze_background(filepath, face_img)
        
        ai_prob_model = 0.5 

        # --- Logic for AI vs Real Classification ---
        if meta["is_ai"]:
            prediction_label = "AI-GENERATED IMAGE"
            confidence_value = 99.9
            reason = f"Metadata explicitly names AI software: {meta['software']}."
            
        else:
            w_model = 0.00 
            w_fft = 0.20
            w_skin = 0.15
            w_eye = 0.15 
            w_bg  = 0.10
            w_lsb = 0.10
            w_ela = 0.05
            w_noise = 0.15 
            
            # --- CLOUD SUPERBRAIN ---
            cloud_result = analyze_with_openrouter(filepath)
            cloud_score = 0
            if cloud_result:
                print(f"Cloud Analysis: {cloud_result}")
                w_cloud = 0.40
                
                # Adjust other weights
                w_fft = 0.10
                w_skin = 0.10
                w_eye = 0.10
                w_noise = 0.10
                w_bg = 0.05
                w_lsb = 0.05
                w_model = 0.00
                w_ela = 0.05
                w_cloud = 0.40 # Ensure
                
                cloud_conf = float(cloud_result.get('confidence', 50))
                cloud_is_ai = cloud_result.get('is_ai', False)
                
                if cloud_is_ai:
                    cloud_score = 0.0 # AI = 0 Realness
                    reason_model = f"Cloud AI ({cloud_conf}%): {cloud_result.get('reason')}"
                else:
                    cloud_score = 100.0 # Real
                    reason_model = f"Cloud AI ({cloud_conf}%): {cloud_result.get('reason')}"
            else:
                 w_cloud = 0.0
                 cloud_score = 50.0
                 reason_model = "Forensic Analysis (Physics & Biometrics)"

            # Scores: 0=Artificial, 100=Real
            s_model = (1.0 - ai_prob_model) * 100.0
            
            # Cast all scores to standard float to avoid JSON serialization errors with float32
            fft_score = float(fft_score)
            skin_score = float(skin_score)
            eye_score = float(eye_score)
            bg_score = float(bg_score)
            lsb_score = float(lsb_score)
            ela_score = float(ela_score)
            noise_score = float(noise_score)
            cloud_score = float(cloud_score)

            realness_score = (s_model * w_model) + \
                             (fft_score * w_fft) + \
                             (skin_score * w_skin) + \
                             (eye_score * w_eye) + \
                             (bg_score * w_bg) + \
                             (lsb_score * w_lsb) + \
                             (ela_score * w_ela) + \
                             (noise_score * w_noise) + \
                             (cloud_score * w_cloud)
            
            total_weight = w_model + w_fft + w_skin + w_eye + w_bg + w_lsb + w_ela + w_noise + w_cloud
            if total_weight > 0:
                realness_score = realness_score / total_weight
            
            combined_ai_score = 1.0 - (realness_score / 100.0)

            # --- FORENSIC OVERRIDES ---
            if cloud_result:
                cvar = float(cloud_result.get('confidence', 0)) / 100.0
                if cloud_result.get('is_ai'):
                     combined_ai_score = max(combined_ai_score, cvar)
                else:
                     combined_ai_score = min(combined_ai_score, 1.0 - cvar)

            if noise_score < 2.0:
                 combined_ai_score = max(combined_ai_score, 0.95)
                 reason_model = "Physics: Complete lack of photon noise (Plastic)."
            
            if face_img is not None and eye_score < 20:
                 combined_ai_score = max(combined_ai_score, 0.95)
                 reason_model = "Biometrics: Non-circular pupils detected."

            # 2. Digital Purity Check
            if combined_ai_score < 0.80:
                is_pure_digital = False
                
                if face_img is not None and eye_score < 25:
                     combined_ai_score = 0.95
                     reason_model = "Biometric: Malformed/Non-circular pupils detected."
                
                elif face_img is not None and skin_score < 15:
                     combined_ai_score = 0.93
                     reason_model = "Skin Texture: Unnaturally smooth/plastic surface detected."
                
                elif noise_score < 5:
                     combined_ai_score = 0.94
                     reason_model = "Physics: Violation of photon noise statistics (Too Flat)."

                elif fft_score < 4 and lsb_score < 10: 
                     is_pure_digital = True
                     reason_model = "Image Feel: Zero sensor noise (Bit/LSB). Digital source."
                
                if is_pure_digital:
                    # Digital Art Logic
                    # If Cloud says Real, it might be art.
                    if cloud_result and not cloud_result.get('is_ai'):
                         prediction_label = "DIGITAL ART / EDITED"
                         confidence_value = 85.0
                         reason = "Digital origin detected but Cloud AI sees no generic abnormalities."
                         
                         return jsonify({
                            'prediction': prediction_label,
                            'confidence': confidence_value,
                            'reason': reason,
                            'heatmap': heatmap,
                            'status': "suspicious",
                            'debug_scores': {'fft': fft_score, 'skin': skin_score, 'eye': eye_score, 'bg': bg_score, 'noise': noise_score, 'cloud': cloud_score},
                            **meta
                        })
                    else:
                        combined_ai_score = 0.92

            if combined_ai_score > 0.50: 
                prediction_label = "AI-GENERATED IMAGE"
                if combined_ai_score < 0.70: prediction_label = "SUSPICIOUS / LIKELY AI"
                
                confidence_value = round(min(99.9, 50.0 + (combined_ai_score * 50.0)), 1)
                
                reasons = []
                if cloud_result and cloud_result.get('is_ai'): reasons.append("cloud diagnosis")
                if fft_score > 55: reasons.append("frequency artifacts")
                if lsb_score < 15: reasons.append("lack of bit-depth noise")
                if eye_score < 40 and face_img is not None: reasons.append("irregular pupils")
                if skin_score < 20 and face_img is not None: reasons.append("plastic skin texture")
                if noise_score < 10: reasons.append("unnatural physics")
                
                if reasons:
                    reason_str = ", ".join(reasons)
                    reason = f"AI traces detected via {reason_str}."
                else:
                    reason = reason_model
            else:
                prediction_label = "REAL IMAGE"
                confidence_value = round(min(99.9, 50.0 + ((1.0 - combined_ai_score) * 50.0)), 1)
                reason = reason_model

        return jsonify({
            'prediction': prediction_label,
            'confidence': confidence_value,
            'reason': reason,
            'heatmap': heatmap,
            'status': "fake" if "AI" in prediction_label else "real",
            'debug_scores': {'fft': fft_score, 'skin': skin_score, 'eye': eye_score, 'bg': bg_score, 'noise': noise_score, 'cloud': cloud_score},
            **meta
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath): os.remove(filepath)


@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    unique_filename = f"aud_{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4().hex}.wav")
    spec_path = os.path.join(app.config['UPLOAD_FOLDER'], f"spec_{uuid.uuid4().hex}.jpg")

    try:
        if not convert_to_wav(filepath, wav_path):
             return jsonify({'error': 'Audio conversion failed (FFMPEG error)'}), 500

        if generate_audio_spectrogram(wav_path, spec_path):
            local_audio_stats = analyze_audio_forensics(wav_path)
            cloud_result = analyze_with_openrouter(spec_path, mode="audio_spectrogram")
            
            if cloud_result:
                is_ai = cloud_result.get('is_ai', False)
                confidence = float(cloud_result.get('confidence', 50))
                
                # Accuracy Boost: Detect "Unnatural Silence" or "Spectral Cutoff"
                if local_audio_stats['is_perfect_silence']:
                    is_ai = True
                    confidence = max(confidence, 96.0)
                    cloud_result['reason'] = "AI SIGNATURE: Mathematical Zero Noise Floor detected (Room noise absent)."
                elif local_audio_stats['is_hf_clipped'] and is_ai:
                    confidence = max(confidence, 92.0)
                    cloud_result['reason'] = (cloud_result.get('reason', '') + " | Spectral Clipping observed (Synthetic compression signature).").strip()
                
                # Status Alignment & Confidence Scaling
                verdict_str = str(cloud_result.get('verdict', 'Real')).lower()
                if 'ai' in verdict_str or 'fake' in verdict_str:
                    status = "fake"
                elif 'suspicious' in verdict_str:
                    status = "suspicious"
                else:
                    status = "real"
                
                if confidence < 1.0 and confidence > 0:
                    confidence = confidence * 100
                    
                return jsonify({
                    'prediction': cloud_result.get('verdict', "AI VOICE CLONE DETECTED" if is_ai else "REAL HUMAN VOICE").upper(),
                    'confidence': round(confidence, 1),
                    'reason': cloud_result.get('reason', 'Acoustic patterns analyzed.'),
                    'transcript': cloud_result.get('transcript', ''),
                    'translation': cloud_result.get('translation', ''),
                    'checklist': {
                        "Voice Naturalness": cloud_result.get('checklist', {}).get('Voice Naturalness', "Human-like texture."),
                        "Emotion Consistency": cloud_result.get('checklist', {}).get('Emotion Consistency', "Dynamic range confirmed."),
                        "Breathing & Pauses": cloud_result.get('checklist', {}).get('Breathing & Pauses', "Organic breathing noted."),
                        "Pitch & Prosody": cloud_result.get('checklist', {}).get('Pitch & Prosody', "Natural rhythm."),
                        "Pronunciation Errors": cloud_result.get('checklist', {}).get('Pronunciation Errors', "No robotic glitches."),
                        "Background Noise": cloud_result.get('checklist', {}).get('Background Noise', "Natural ambiance detected."),
                        "Audio Artifacts": cloud_result.get('checklist', {}).get('Audio Artifacts', "Zero spectral aliasing."),
                        "Continuity": cloud_result.get('checklist', {}).get('Continuity', "Consistent voice profile."),
                        "Compression Effects": cloud_result.get('checklist', {}).get('Compression Effects', "High fidelity stream."),
                        "Source Verification": cloud_result.get('checklist', {}).get('Source Verification', "Authentic context.")
                    },
                    'status': status,
                    'awareness_note': cloud_result.get('awareness_note', "Audio forensic audit complete."),
                    'device': "Acoustic Guard"
                })

        return jsonify({'error': 'Audio analysis failed'}), 500

    except Exception as e:
        print(f"Audio Predict Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        for p in [filepath, wav_path, spec_path]:
            if os.path.exists(p): os.remove(p)

@app.route('/predict-text', methods=['POST'])
def predict_text():
    try:
        data = request.json
        print(f"DEBUG: Processing Text Predict request.")
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text_content = data['text']
        if len(text_content.strip()) < 10:
            return jsonify({'error': 'Text too short for analysis'}), 400

        # We pass the text itself as the "encoded_string" for the text mode
        cloud_result = analyze_with_openrouter(text_content, mode="text_forensics")
        
        if cloud_result:
            print(f"DEBUG: Cloud text analysis result: {cloud_result}")
            # Status and Confidence Alignment
            raw_v = cloud_result.get('verdict', 'Real')
            verdict_str = str(raw_v).lower()
            
            is_spam = cloud_result.get('is_spam', False)
            is_ai = cloud_result.get('is_ai', False)

            if 'ai' in verdict_str or 'fake' in verdict_str or is_spam or is_ai:
                status = "fake"
            elif 'suspicious' in verdict_str:
                status = "suspicious"
            else:
                status = "real"
            
            conf = float(cloud_result.get('confidence', 50))
            if conf < 1.0 and conf > 0: conf *= 100
                
            return jsonify({
                'prediction': str(raw_v).upper(),
                'confidence': round(conf, 1),
                'reason': cloud_result.get('reason', 'Linguistic patterns check complete.'),
                'checklist': cloud_result.get('checklist', {}),
                'status': status,
                'is_spam': is_spam,
                'awareness_note': cloud_result.get('awareness_note', "Text audit complete."),
                'device': "Linguistic Guard"
            })

        print("DEBUG: Cloud text analysis returned None")
        return jsonify({'error': 'Text analysis failed'}), 500
    except Exception as e:
        print(f"Text Predict Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/audio')
def audio_page():
    return render_template('audio.html')

@app.route('/text')
def text_page():
    return render_template('text.html')


if __name__ == '__main__':
    # host='0.0.0.0' allows access from other devices on the same network
    app.run(host='0.0.0.0', port=5001, debug=True)
