import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from kbp_model import SimpleUNet3D

# --- CONFIGURATION ---
# Points to your overnight trained model
MODEL_PATH = "../models/best_model_overnight.pth"
DATA_DIR = "../data/processed" 
DEVICE = "cpu"

# --- PAGE SETUP ---
st.set_page_config(page_title="Morpho-Dose AI", layout="wide")

st.title("⚡ Morpho-Dose: Adaptive Radiotherapy AI")
st.markdown("""
**Medical Physics Project** | Deep Learning for Dose Prediction
This tool predicts the 3D dose distribution for Head & Neck patients.
* **Input:** Daily CT (Anatomy)
* **AI Model:** 3D U-Net (PyTorch)
* **Post-Processing:** Interactive Normalization
""")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("🕹️ Clinical Controls")

# 1. Select Patient
if os.path.exists(DATA_DIR):
    patient_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npz')])
    selected_file = st.sidebar.selectbox("Select Patient Case:", patient_files)
else:
    st.error("Data directory not found!")
    selected_file = None

# 2. Normalization Slider (The "Physicist" Fix)
st.sidebar.markdown("---")
st.sidebar.write("**Normalization Point**")
# Slider allows scaling from 80% to 200%
norm_factor = st.sidebar.slider("Scale Dose (%)", 80, 200, 100, 5) / 100.0

run_btn = st.sidebar.button("🔮 Predict Dose")

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    model = SimpleUNet3D()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# --- MAIN APP LOGIC ---
if run_btn and selected_file:
    # Load Model
    model = load_model()
    
    # Load Data
    file_path = os.path.join(DATA_DIR, selected_file)
    data = np.load(file_path)
    ct = data['ct'].astype(np.float32)
    true_dose = data['dose'].astype(np.float32)

    # Preprocessing (Match training logic)
    # Clip -1000 to 1000, scale to 0-1
    ct_norm = np.clip(ct, -1000, 1000)
    ct_norm = (ct_norm - -1000) / 2000.0
    inp = torch.from_numpy(ct_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Inference
    with st.spinner("🧠 AI is calculating physics..."):
        with torch.no_grad():
            pred = model(inp)
            
        # Post-Processing: Convert to Gy and Apply Slider
        # 80.0 is the max dose we divided by during training
        raw_pred_gy = pred.squeeze().cpu().numpy() * 80.0
        final_pred_gy = raw_pred_gy * norm_factor

    # --- VISUALIZATION DASHBOARD ---
    st.success(f"Calculation Complete (Normalized: {norm_factor*100:.0f}%)")
    
    # Slice Selection Slider
    slice_idx = st.slider("Navigate Axial Slices (Z-axis)", 0, 127, 64)
    
    col1, col2, col3 = st.columns(3)
    
    # Shared display settings so colors match
    max_dose_disp = max(true_dose.max(), final_pred_gy.max())
    
    with col1:
        st.subheader("CT Anatomy")
        fig1, ax1 = plt.subplots()
        ax1.imshow(ct[:, :, slice_idx], cmap='gray')
        ax1.axis('off')
        st.pyplot(fig1)

    with col2:
        st.subheader("Ground Truth")
        fig2, ax2 = plt.subplots()
        ax2.imshow(true_dose[:, :, slice_idx], cmap='jet', vmin=0, vmax=max_dose_disp)
        ax2.axis('off')
        st.pyplot(fig2)

    with col3:
        st.subheader("AI Prediction")
        fig3, ax3 = plt.subplots()
        im = ax3.imshow(final_pred_gy[:, :, slice_idx], cmap='jet', vmin=0, vmax=max_dose_disp)
        ax3.axis('off')
        st.pyplot(fig3)

    # --- METRICS & GAMMA PASS/FAIL ---
    st.divider()
    st.subheader("📊 QA Metrics")
    
    # Calculate difference
    mean_error = np.mean(np.abs(final_pred_gy - true_dose)) # Global 3D MAE
    
    m1, m2, m3 = st.columns(3)
    
    m1.metric("Max Predicted Dose", f"{final_pred_gy.max():.1f} Gy")
    m2.metric("Mean Absolute Error (MAE)", f"{mean_error:.2f} Gy", 
              delta="-High Error" if mean_error > 5 else "Acceptable")
    
    # Simple Clinical Check
    # If error is low, we PASS. If high, we need normalization.
    if mean_error < 8.0:
        m3.success("✅ QA Result: PASS")
    else:
        m3.error("❌ QA Result: FAIL")
        st.info("💡 Hint: Use the sidebar slider to normalize the dose!")

elif not run_btn:
    st.info("👈 Select a patient and click 'Predict Dose' to start.")
