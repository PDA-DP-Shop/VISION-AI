# VISION AI - Deepfake Detection & Forensic Analysis 🛡️

**Imagine Cup 2026 Submission** | **Powered by Microsoft AI**

![Status](https://img.shields.io/badge/Status-Active-success) 
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) 
![ONNX Runtime](https://img.shields.io/badge/Microsoft-ONNX%20Runtime-blue?logo=microsoft) 
![Microsoft Phi-3](https://img.shields.io/badge/AI-Microsoft%20Phi--3-green?logo=microsoft)

## 💡 Elevator Pitch
**VISION AI** is an advanced multi-modal forensic platform designed to restore trust in digital media. By combining **Computer Vision (Deep Learning)** with **Microsoft Phi-3 (Linguistic Analysis)** and optimizing inference with **Microsoft ONNX Runtime**, we detect AI-generated deepfakes, manipulated documents, and synthetic audio with enterprise-grade precision.

---

## 🚀 Key Features

### 1. 👁️ Visual Forensics (Deepfake Detection)
*   **Hybrid Analysis**: Combines EfficientNet (Spatial) and FFT (Frequency) analysis to detect GAN/Diffusion artifacts.
*   **Microsoft ONNX Runtime Integration**: 
    *   Optimized inference engine for cross-platform compatibility.
    *   Accelerated model performance using `Microsoft.ML.OnnxRuntime`.

### 2. 📝 Text Forensics (Powered by Microsoft Phi-3)
*   **Linguistic Audit**: Uses **Microsoft Phi-3** (via OpenRouter) to scan text metadata and content.
*   **Pattern Recognition**: Detects "LLM-speak", distinct phrasing, and structural anomalies typical of AI-generated text.

### 3. 📄 Document Verification
*   **Error Level Analysis (ELA)**: Detects digital splicing and compression inconsistencies in ID cards and contracts.
*   **Metadata Forensics**: Extracts EXIF data to identify editing software (Photoshop, GIMP) vs. original camera timestamps.

### 4. 🔊 Audio Spectrum Analysis
*   **Forensic Spectrograms**: Visualizes high-frequency cutoffs typical of synthetic voice cloning (ElevenLabs, etc.).
*   **Noise Floor Detection**: Identifies unnatural silence and lack of background thermal noise.

---

## 🛠️ Microsoft Technology Integration

This project proudly leverages the Microsoft ecosystem to solve the global challenge of misinformation:

| Technology | Usage in VISION AI |
| :--- | :--- |
| **Microsoft ONNX Runtime** | We convert our robust Keras models to `.onnx` format using `tf2onnx` and run heavily optimized inference. This ensures our solution can run on Edge devices and low-power hardware. |
| **Microsoft Phi-3** | We utilize the **Microsoft Phi-3-Mini-128k-Instruct** model (via API) to perform lightweight, highly intelligent reasoning on metadata and textual content to flag AI generation. |
| **GitHub** | The entire codebase is version-controlled and hosted on Microsoft GitHub for open-source collaboration. |

---

## 💻 Tech Stack

*   **Core**: Python 3.9+
*   **Web Framework**: Flask (production-ready)
*   **AI & ML**: TensorFlow, **ONNX Runtime**, OpenCV, NumPy
*   **LLM Integration**: **Microsoft Phi-3** (via OpenRouter API)
*   **Frontend**: HTML5, CSS3, JavaScript (Responsive Dashboard)

---

## ⚙️ Installation & Local Setup

### Prerequisites
*   Python 3.9+
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/PDA-DP-Shop/VISION-AI.git
cd VISION-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Models (Important)
Due to GitHub size limits, the large model files are excluded.
*   **Option A (Recommended)**: Download the pre-trained `xception_deepfake.h5` model and place it in the `model/` directory.
*   **Option B (Optimize)**: Run the ONNX conversion script to generate the optimized model:
    ```bash
    python convert_to_onnx.py
    ```

### 4. Run the Application
```bash
python app.py
```
*   Access the dashboard at: `http://localhost:5001`

---

## 📸 Usage Guide
1.  **Navigate to "Scan"**: Upload an image, video, or document.
2.  **View Results**:
    *   **Fake %**: Probability of AI generation.
    *   **Heatmaps**: Visual indicators of manipulated regions (ELA).
    *   **Metadata Report**: detailed breakdown of the file source.
3.  **Video Analysis**: Upload a video to scan it frame-by-frame for temporal inconsistencies.

---

## 📜 License
This project is licensed under the Apache 2.0 License.

---
*Built for Microsoft Imagine Cup 2026*
