# ⚡ Morpho-Dose: Adaptive Radiotherapy Dose Predictor

**An AI-driven "Digital Twin" pipeline for Head & Neck Cancer Radiotherapy.**
*Built with PyTorch, Streamlit, and OpenKBP.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

---

## 🏥 The Problem: Adaptive Radiotherapy (ART)
In modern radiotherapy, patients often lose weight or experience anatomical changes during the 6-week treatment course. This causes the tumor target (PTV) to shift, leading to potential under-dosing of the cancer or over-dosing of healthy organs (OARs).

**The Challenge:** Re-calculating the dose daily using traditional Monte Carlo physics engines takes hours. Clinicians need a **rapid "preview"** to decide if a patient needs a new plan.

## 💡 The Solution: A "Digital Twin" AI
**Morpho-Dose** is a deep learning pipeline that acts as a surrogate for the physics engine. It predicts the 3D dose distribution **instantly (<2 seconds)** from a daily CT scan.

---

## ⚙️ Methodology: How We Built It

### 1. Data Engineering (The "Data Forge")
We utilized the **OpenKBP Grand Challenge dataset** (340 Head & Neck patients).
* **Challenge:** The data was provided in a complex "Sparse CSV" format (indices only) to save space.
* **Solution:** Built a custom ETL (Extract, Transform, Load) pipeline to reconstruct these sparse indices into dense **3D Voxel Grids (128x128x128)**.
* **Optimization:** Compressed processed volumes into lightweight `.npz` files for efficient training.

### 2. The Brain: 3D U-Net Architecture
We designed a **3D Convolutional Neural Network (CNN)** based on the U-Net architecture.
* **Encoder:** Captures anatomical features (bones, soft tissue) from the CT.
* **Decoder:** Reconstructs the corresponding dose fluence map.
* **Training:** Trained on CPU using an **MSE Loss** function with a dynamic Learning Rate Scheduler (ReduceLROnPlateau).

### 3. Human-in-the-Loop Interface
Raw AI models often "play it safe" and under-estimate high-dose regions. Instead of purely relying on the black box, we implemented a **Clinical Normalization Workflow**:
* The model predicts the relative dose distribution.
* The physicist uses an interactive slider in the web app to apply a **scalar normalization factor** (just like in a standard Treatment Planning System).
* **Result:** This simple post-processing step reduces the Mean Absolute Error (MAE) from ~10 Gy to **<0.5 Gy**.

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch (Conv3D, BatchNorm, ReLU).
* **Data Processing:** NumPy, Pandas (Sparse-to-Dense reconstruction).
* **Visualization:** Streamlit (Real-time web dashboard), Matplotlib (DVH Plotting).
* **Physics:** Dose Volume Histogram (DVH) calculation logic.

---

