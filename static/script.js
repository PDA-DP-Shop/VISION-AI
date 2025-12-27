const BASE_URL = "";

/**
 * Shared function to handle all forensic analysis requests
 * @param {string} endpoint - The URL to fetch (/predict or /predict-doc)
 */

/**
 * Shared function to handle all forensic analysis requests
 * Updated to support BATCH processing.
 */
const performAnalysis = async (endpoint, btnId = 'runBtn') => {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;

    if (!files || files.length === 0) {
        alert("Please select at least one file.");
        return;
    }

    const btn = document.getElementById(btnId);

    // UI Reset
    document.getElementById('resultContent').classList.remove('hidden');
    document.getElementById('loader').classList.add('hidden'); // We use batch progress instead
    if (btn) btn.classList.add('hidden');

    const placeholder = document.getElementById('placeholderText');
    if (placeholder) placeholder.classList.add('hidden');

    const resultPanel = document.getElementById('resultPanel');
    if (resultPanel) resultPanel.style.opacity = "1.0";

    // Setup Batch UI
    const batchProgress = document.getElementById('batchProgress');
    const batchBar = document.getElementById('batchBar');
    const batchStatus = document.getElementById('batchStatus');

    if (batchProgress) {
        if (files.length > 1) {
            batchProgress.classList.remove('hidden');
        } else {
            batchProgress.classList.add('hidden');
        }
    }

    // Always show loader if not batch or if batch UI missing
    if (files.length <= 1 || !batchProgress) {
        const loader = document.getElementById('loader');
        if (loader) loader.classList.remove('hidden');
    }

    // Process Files
    let processed = 0;
    let fakeCount = 0;
    let realCount = 0;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];

        // Update Status
        if (files.length > 1 && batchBar && batchStatus) {
            batchBar.style.width = `${((i) / files.length) * 100}%`;
            batchStatus.innerText = `Processing ${i + 1}/${files.length}: ${file.name}`;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${BASE_URL}${endpoint}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Server Error");
            const data = await response.json();

            // Stats
            const isFake = data.status === "fake" || data.prediction.includes('AI');
            if (isFake) fakeCount++;
            else realCount++;

            // If it's the LAST file, or Single file, update main UI
            if (i === files.length - 1) {
                updateUI(data);
                if (files.length > 1) {
                    // Append Batch Summary to Prediction Text
                    setTimeout(() => {
                        const predText = document.getElementById('predictionText');
                        predText.innerHTML += `<br><br><span style="color:#2563eb">Batch Complete: Found ${fakeCount} Fake/AI images out of ${files.length}.</span>`;
                    }, 500);
                }
            }

        } catch (err) {
            console.error(err);
        }

        processed++;
    }

    // Finalize
    if (batchBar) batchBar.style.width = "100%";
    if (batchStatus) batchStatus.innerText = "Completed!";
    if (btn) btn.classList.remove('hidden');
    document.getElementById('loader').classList.add('hidden');
};

/**
 * Updates the UI with the JSON response from Flask
 */
const updateUI = (data) => {
    // ... (Keep existing UI update logic for single result display)
    document.getElementById('resultContent').classList.remove('hidden');

    // 1. Core Elements
    const predictionText = document.getElementById('predictionText');
    const predictionBadge = document.getElementById('predictionBadge');
    const confNum = document.getElementById('confNum');
    const reasonText = document.getElementById('reasonText');
    const bar = document.getElementById('barFill');
    const metaSoft = document.getElementById('metaSoft');
    const metaDev = document.getElementById('metaDev');

    // Classification Logic & Styling
    let statusClass = "UNKNOWN";
    let statusText = data.prediction;

    // Normalize status
    const isFake = data.status === "fake" || statusText.includes('FAKE') || statusText.includes('AI') || statusText.includes('MODIFIED') || data.is_spam;
    const isReal = !isFake && (data.status === "real" || statusText.includes('REAL'));

    if (isFake) {
        statusClass = "ALTERED";
        if (predictionBadge) {
            predictionBadge.innerText = data.is_spam ? "FAKE / SPAM" : "FAKE / AI";
            predictionBadge.style.background = "#fee2e2";
            predictionBadge.style.color = "#dc2626";
        }
    } else if (isReal) {
        statusClass = "AUTHENTIC";
        if (predictionBadge) {
            predictionBadge.innerText = "REAL";
            predictionBadge.style.background = "#d1fae5";
            predictionBadge.style.color = "#059669";
        }
    } else {
        if (predictionBadge) {
            predictionBadge.innerText = "SUSPICIOUS";
            predictionBadge.style.background = "#ffedd5";
            predictionBadge.style.color = "#c2410c";
        }
    }

    // Update Text & Badge Container
    if (predictionText) predictionText.innerText = data.prediction;

    // Update Confidence Label
    const scoreLabel = document.getElementById('scoreLabel');
    if (scoreLabel) {
        if (isReal) scoreLabel.innerText = "Authenticity Score";
        else scoreLabel.innerText = data.is_spam ? "Spam Risk probability" : "AI Probability";
    }

    // Update Confidence
    if (confNum) confNum.innerText = data.confidence + "%";

    // Update Reason
    if (reasonText) reasonText.innerText = data.reason;

    // Video Specific Detailed Results (10-Point Checklist)
    const forensicReport = document.getElementById('forensicReport');
    const checklistGrid = document.getElementById('checklistGrid');
    const awarenessBox = document.getElementById('awarenessBox');
    const awarenessText = document.getElementById('awarenessText');
    const compressionWarning = document.getElementById('compressionWarning');
    const localReason = document.getElementById('localReason');

    const vettingBadge = document.getElementById('vettingBadge');

    if (data.checklist && checklistGrid) {
        forensicReport.classList.remove('hidden');
        checklistGrid.innerHTML = "";

        if (data.vetted_by && vettingBadge) {
            vettingBadge.innerText = "Audited by: " + data.vetted_by.split('/')[1] || data.vetted_by;
        }
        Object.entries(data.checklist).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.style.padding = "10px";
            item.style.background = "#f1f5f9";
            item.style.borderRadius = "6px";
            item.style.fontSize = "0.9rem";
            item.innerHTML = `<span style="font-weight:700; color:var(--primary);">${key}:</span> <span style="color:#475569;">${value}</span>`;
            checklistGrid.appendChild(item);
        });

        if (awarenessText) awarenessText.innerText = data.awareness_note || "Analysis complete.";
        if (compressionWarning) compressionWarning.innerText = data.compression_warning || "";

        // Render Transcript & Translation
        const transcriptBox = document.getElementById('transcriptBox');
        const transcriptText = document.getElementById('transcriptText');
        const translationText = document.getElementById('translationText');

        if (transcriptBox && (data.transcript || data.translation)) {
            transcriptBox.classList.remove('hidden');
            if (transcriptText) transcriptText.innerText = data.transcript ? `"${data.transcript}"` : "";
            if (translationText) translationText.innerText = data.translation ? `Translation: ${data.translation}` : "";
        } else if (transcriptBox) {
            transcriptBox.classList.add('hidden');
        }
    } else if (forensicReport) {
        forensicReport.classList.add('hidden');
    }

    if (data.local_insight && localReason) {
        localReason.innerText = data.local_insight;
    }

    // --- RENDER FORENSIC BREAKDOWN (For Images / Legacy) ---
    const scoreList = document.getElementById('scoreList');
    if (scoreList && data.debug_scores) {
        scoreList.innerHTML = ""; // Clear previous

        const scores = data.debug_scores;

        // Helper to create list item
        const addItem = (label, score, type) => {
            let isPass = false;
            let text = "";

            if (label === "Cloud AI Analysis") {
                if (window.location.pathname.includes('doc')) {
                    isPass = score > 50;
                    label = "3. Layout Audit";
                    text = isPass ? "Layout matches known official reference templates." : "Significant deviation from official reference templates.";
                } else {
                    isPass = score > 50;
                    text = isPass ? "No anomalies detected by Cloud LLM." : "AI artifacts identified (Anatomy/Text).";
                    if (score === 50) { isPass = true; text = "Cloud Analysis inconclusive/offline."; }
                }
            }

            else if (label === "Photon Physics") {
                // Digital Purity
                if (window.location.pathname.includes('doc')) {
                    isPass = score > 40;
                    label = "4. Digital Purity";
                    text = isPass ? "Background texture is consistent with scanned media." : "Mathematically pure digital origin detected.";
                } else {
                    isPass = score > 10;
                    text = isPass ? "Sensor noise patterns consistent with real optics." : "Zero sensor noise detected (Plastic/Synthetic).";
                }
            }

            else if (label === "Biometrics (Eyes)") {
                if (window.location.pathname.includes('doc')) return;
                isPass = score > 40;
                text = isPass ? "Pupils are circular and synonymous." : "Irregular or asymmetric pupil shapes detected.";
            }

            else if (label === "Skin Texture") {
                // Mapped to QR Score
                if (window.location.pathname.includes('doc')) {
                    isPass = score > 60;
                    label = "5. Security Codes (QR)";
                    text = isPass ? "Machine-readable data pattern found." : "No valid data patterns (QR/Bar) detected.";
                } else {
                    isPass = score > 20;
                    text = isPass ? "Natural pore/hair texture variation found." : "Surface is mathematically smooth (Plasticity).";
                }
            }

            else if (label === "Frequency Analysis") {
                if (window.location.pathname.includes('doc')) {
                    isPass = score > 50; // New logic: Higher is real
                    label = "1. Sensor Noise (FFT)";
                    text = isPass ? "Random frequency noise consistent with scans." : "Synthetic frequency artifacts or grid detected.";
                } else {
                    isPass = score < 60; // Old logic for image mode
                    text = isPass ? "No grid artifacts detected." : "High-frequency checkerboard artifacts found.";
                }
            }

            else if (label === "Background Logic") {
                if (window.location.pathname.includes('doc')) {
                    isPass = score > 50;
                    label = "2. Compression State (ELA)";
                    text = isPass ? "Uniform error levels across the document." : "Non-uniform compression areas (Signs of local editing).";
                } else {
                    isPass = score > 20;
                    text = isPass ? "Depth of field looks natural." : "Inconsistent background blur checks.";
                }
            }

            const icon = isPass ? "✅" : "❌";
            const color = isPass ? "#059669" : "#dc2626";

            const li = document.createElement('li');
            li.style.marginBottom = "8px";
            li.innerHTML = `<span style="font-weight:bold; color:${color}">${icon} ${label}:</span> ${text} <span style="font-size:0.8em; color:#94a3b8">(${Number(score).toFixed(1)})</span>`;
            scoreList.appendChild(li);
        };

        // Add items strictly
        if (scores.cloud !== undefined) addItem("Cloud AI Analysis", scores.cloud);
        if (scores.noise !== undefined) addItem("Photon Physics", scores.noise);
        if (scores.eye !== undefined) addItem("Biometrics (Eyes)", scores.eye);
        if (scores.skin !== undefined) addItem("Skin Texture", scores.skin);
        if (scores.fft !== undefined) addItem("Frequency Analysis", scores.fft);
        if (scores.bg !== undefined) addItem("Background Logic", scores.bg);
    }

    // Update Progress Bar
    if (bar) {
        bar.style.width = "0%";
        bar.style.backgroundColor = isFake ? '#dc2626' : (isReal ? '#059669' : '#c2410c');
        setTimeout(() => {
            bar.style.width = data.confidence + "%";
        }, 100);
    }

    // Metadata
    if (metaDev) metaDev.innerText = data.device || "N/A";
    if (metaSoft) metaSoft.innerText = data.source_category || "Unknown";
};


// --- Event Listeners & Drag/Drop ---

const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const previewContainer = document.getElementById('previewContainer');
const uploadText = document.getElementById('uploadText');
const imagePreview = document.getElementById('imagePreview');

// Handle File Selection (Click & Drag)
const handleFileSelect = (files) => {
    // Modified to handle multiple files preview
    if (files.length > 0) {
        const file = files[0];
        // Only preview the first one for simplicity to avoid UI clutter
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            if (files.length > 1) {
                // Overlay count
                uploadText.classList.remove('hidden');
                document.getElementById('fileName').innerText = `${files.length} Files Selected`;
                previewContainer.classList.add('hidden'); // Hide simple preview if batch
            } else {
                previewContainer.classList.remove('hidden');
                uploadText.classList.add('hidden');
            }
        };
        reader.readAsDataURL(file);
    }
};

if (fileInput) {
    fileInput.onchange = (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files);
    };
}

// Drag & Drop Visuals
if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--primary)";
        dropZone.style.backgroundColor = "#eff6ff";
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "#cbd5e1";
        dropZone.style.backgroundColor = "#f8fafc";
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "#cbd5e1";
        dropZone.style.backgroundColor = "#f8fafc";

        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files; // Assign to input
            handleFileSelect(e.dataTransfer.files);
        }
    });
}

// Button Listeners
const runBtn = document.getElementById('runBtn');
if (runBtn) runBtn.onclick = () => performAnalysis('/predict', 'runBtn');

const docBtn = document.getElementById('docBtn');
if (docBtn) docBtn.onclick = () => performAnalysis('/predict-doc', 'docBtn');

// Portal Bridge Handler
const openGovPortal = () => {
    const selector = document.getElementById('docTypeSelector');
    const url = selector.value;
    if (url) {
        window.open(url, '_blank');
    } else {
        alert("Please select a document type to verify.");
    }
};