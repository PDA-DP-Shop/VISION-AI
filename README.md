# Sentry AI - Deepfake Detection & Forensic Analysis 🛡️

**Sentry AI** represents the next generation of media forensics. It is a full-stack Enterprise Security platform designed to detect **AI-generated images**, **manipulated documents**, and **synthetic media** with high precision.

![Sentry AI Hero](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) ![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

## 🚀 Key Features

### 1. Hybrid Forensic Engine
Unlike simple classifiers, Sentry AI combines **three distinct layers** of analysis:
*   **Deep Learning (EfficientNetB0)**: Detects visual patterns and artifacts characteristic of GANs and Diffusion models (Midjourney, Stable Diffusion).
*   **Frequency Analysis (FFT)**: Performs Fast Fourier Transform to analyze the spectral domain, detecting the "unnatural smoothness" or grid artifacts of AI generation.
*   **Error Level Analysis (ELA)**: specific for documents, it detects compression inconsistencies that occur when an image has been digitally altered or spliced.

### 2. Document Verification System
A specialized module (`/predict-doc`) for verifying high-stakes documents (IDs, Certificates, Contracts):
*   Distinguishes between **Original Camera**, **Scans**, and **Screen Captures**.
*   Detects if text has been digitally modified using ELA.
*   Verifies "Digital Purity" effectively distinguishing real e-documents from generated fakes.

### 3. Enterprise-Grade UI
*   Modern, responsive interface built with HTML5/CSS3 and Flask.
*   Interactive heatmaps for visual verification.
*   Real-time confidence scoring.

---

## 🛠️ Installation & Local Setup

### Prerequisites
*   Python 3.9+
*   Git

### Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/sentry-ai.git
    cd sentry-ai
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python app.py
    ```
    The app will start at `http://127.0.0.1:5001`.

---

## ☁️ Deployment (Hugging Face Spaces)

This project is optimized for deployment on **Hugging Face Spaces** using Docker.

### How to Deploy
1.  Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2.  Select **Docker** as the SDK.
3.  Upload the contents of this repository.
    *   *Note*: Ensure `sentry_forensic_v3.h5` is uploaded (use Git LFS if pushing via command line).
4.  The `Dockerfile` will automatically install system dependencies (OpenCV/GL libraries) and start the Gunicorn server.

---

## 🧠 Technical Stack

*   **Backend**: Flask (Python)
*   **ML Engine**: TensorFlow, Keras (EfficientNet)
*   **Image Processing**: OpenCV, Pillow (PIL)
*   **Frontend**: HTML5, CSS3, JavaScript (AOS Animation Library)
*   **Server**: Gunicorn (Production), Werkzeug (Dev)

---

## ⚠️ Disclaimer
This tool is intended for forensic research and security verification. While highly accurate, no detection method is 100% foolproof against all current future generative models. Always use multiple verification methods for critical decisions.
