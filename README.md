# ⚡ Morpho-Dose: Adaptive Radiotherapy Dose Predictor

**An AI-driven "Digital Twin" pipeline for Head & Neck Cancer Radiotherapy.**
*Built with PyTorch, Jupyter, and OpenKBP.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🏥 The Problem: Adaptive Radiotherapy (ART)
In modern radiotherapy, patients often lose weight or experience anatomical changes during the 6-week treatment course. This causes the tumor target (PTV) to shift, leading to potential under-dosing of the cancer or over-dosing of healthy organs (OARs).

**The Challenge:** Re-calculating the dose daily using traditional Monte Carlo physics engines takes hours. Clinicians need a **rapid "preview"** to decide if a patient needs a new plan.

## 💡 The Solution: A "Digital Twin" AI
**Morpho-Dose** is a deep learning pipeline that acts as a surrogate for the physics engine. It predicts the 3D dose distribution **instantly (< 2 seconds)** from a daily CT scan, allowing for real-time quality assurance.

---

## ⚙️ Methodology: How I Built It

### 1. Data Forensics & Engineering
I utilized the **OpenKBP dataset** (Head & Neck patients) provided in a raw CSV format.
* **The Fix:** I identified a critical scaling artifact in the raw CT data (values shifted by +1000). I implemented a custom pre-processing pipeline to shift these values back to the Hounsfield Unit (HU) scale (Air = -1000, Bone = +1000) so the model could correctly interpret anatomy.
* **3D Reconstruction:** I engineered a robust loader to transform sparse CSV indices into dense **128x128x128 Voxel Grids**.
* **Dual-Channel Input:** Unlike standard approaches, I fed the model **2 input channels** (CT Anatomy + PTV Mask) to explicitly teach it *where* the target was located.

### 2. The Brain: 3D U-Net Architecture
I designed a **3D Convolutional Neural Network (CNN)** based on the U-Net architecture.
* **Encoder:** Captures anatomical features (bones, soft tissue) via 3D Convolutions and Max Pooling.
* **Decoder:** Reconstructs the detailed dose fluence map using upsampling and skip connections.
* **Two-Phase Training:**
    1.  **Phase 1 (Anatomy):** Trained for 20 epochs using **MSE Loss** to learn general dose shapes.
    2.  **Phase 2 (Refinement):** Fine-tuned for 30 epochs using **L1 Loss** to sharpen high-dose gradients and reduce under-dosing.

### 3. Clinical Normalization Logic
Raw AI models often produce "safe" (blurry) predictions that under-estimate peak dose. I implemented a standard medical physics normalization step:
* The model predicts the relative dose distribution.
* I calculate a **Calibration Factor** by comparing the mean dose in the PTV region against the clinical prescription.
* **Result:** This significantly reduced the D95 error, ensuring the plan meets clinical coverage requirements.

---

## 📊 Clinical Validation Results (N=8 Hold-out)

I evaluated the refined model on a distinct validation set.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Inference Speed** | **< 2.0 sec** | Enables Real-time QA |
| **Global Accuracy (MAE)** | **0.73 Gy** | Excellent precision (< 1 Gy error) |
| **Target Coverage (D95)** | **8.99 Gy** | Good conformality after normalization |
| **Calibration Factor** | **1.14x** | Minimal post-scaling required (14%) |

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch (Conv3D, BatchNorm, L1Loss).
* **Data Processing:** Pandas, NumPy (Sparse-to-Dense voxelization).
* **Visualization:** Matplotlib (Dose color wash & DVH Plotting).
* **Environment:** Jupyter Notebook (Single-file reproducible pipeline).

---
