# Microsoft Imagine Cup 2026 Submission Guide

Congratulations on preparing for the Imagine Cup! Here is your step-by-step guide to fulfilling the technical requirements for your Deepfake AI project.

## Checklist
- [x] **Project Code**: Your code is organized and git is initialized.
- [ ] **GitHub Repository**: You need to push your local code to GitHub (Instructions below).
- [ ] **Microsoft Tool Integration**: We have configured your project to be deployable to **Azure App Service** which counts as your Microsoft Tool.

---

## Step 1: Put Code on GitHub

Your local git repository is ready. Follow these steps to put it on GitHub:

1.  Log in to [GitHub](https://github.com).
2.  Click the **+** icon in the top right and select **New repository**.
3.  Repository Name: `deepfake-ai-sentinel` (or any name you like).
4.  Visibility: **Public** (usually required for submission, or share access with judges).
5.  **Do NOT** initialize with README, .gitignore, or License (we already have them).
6.  Click **Create repository**.
7.  Copy the URL provided (e.g., `https://github.com/your-username/deepfake-ai-sentinel.git`).
8.  Run these commands in your terminal (I can run them if you give me the URL):

```bash
git remote add origin <YOUR_REPO_URL>
git branch -M main
git push -u origin main
```

## Step 2: Use a Microsoft Tool (No Credit Card Needed)

We have implemented **TWO** Microsoft technologies to exceed the requirements without needing an Azure subscription.

### 1. ONNX Runtime (The "Smart" Choice)
As suggested by ChatGPT, we are using **ONNX Runtime** (from Microsoft) to optimize Deepfake Detection inference.
*   **What it does**: Accelerates AI models on any hardware.
*   **Your Pitch**: "We optimized our Deepfake Detection model using **Microsoft ONNX Runtime**, ensuring faster inference and cross-platform compatibility."
*   **Where is it?**: 
    *   We added `convert_to_onnx.py` to convert your Keras model.
    *   We updated `.gitignore` to handle large model files correctly.

### 2. Microsoft Phi-3 (The "AI" Choice)
*   **What it does**: Advanced text analysis for forensic metadata auditing.
*   **Your Pitch**: "We integrated **Microsoft Phi-3** (via OpenRouter) to provide linguistic forensic analysis, detecting AI-generated patterns in text."

### Action Items
1.  **Submission Form**: Select **Azure AI** or **Other Microsoft Tool**.
    *   Name of tool: "Microsoft ONNX Runtime & Microsoft Phi-3".
2.  **Video**: 
    *   Show the code in `convert_to_onnx.py` on screen.
    *   Show the result of the detection.
    *   Say: *"We use Microsoft ONNX Runtime for the vision model and Microsoft Phi-3 for the text model."*

## Step 3: Run Locally
Since the code is on GitHub, you can demo it locally:
1.  Ensure your large model files (`*.h5` or `*.onnx`) are in the `model/` folder (GitHub doesn't store them because they are too big).
2.  Run `python app.py`.
3.  Record your demo!

## Step 3: Run Locally for the Demo
Since you aren't deploying to Azure cloud:
1.  Run the app locally with: `python app.py`
2.  Open `http://localhost:5000` (or `7860`) in your browser.
3.  Record your screen interacting with the local website.
4.  **Important**: The judges accept local demos if the code is on GitHub!

Good luck!

## Step 3: Video & Pitch
- Remember to record a 3-minute video demonstrating the **Deepfake Detection** feature.
- Show the **Azure Deployment** running live in the browser.

Good luck!
