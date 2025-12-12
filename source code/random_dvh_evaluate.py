import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from kbp_model import SimpleUNet3D

# --- CONFIGURATION ---
# Point this to the FOLDER containing all patients (e.g., validation-pats)
VALIDATION_DIR = r"C:\Users\sajid\OneDrive\Desktop\KBP Dataset\archive\provided-data\validation-pats"
MODEL_PATH = "../models/best_model_overnight.pth"
DEVICE = "cpu"

def load_sparse_volume(csv_path, shape=(128, 128, 128), fill_value=0, dtype=np.float32):
    """ Robust loader for sparse CSVs. """
    if not os.path.exists(csv_path):
        return None
        
    try:
        df = pd.read_csv(csv_path)
        indices = df.iloc[:, 0].values.astype(int)
        
        if df.shape[1] > 1 and not df.iloc[:, 1].isnull().all():
            values = df.iloc[:, 1].values.astype(dtype)
        else:
            values = np.ones(len(indices), dtype=dtype)
            
        total_voxels = shape[0] * shape[1] * shape[2]
        arr_flat = np.full(total_voxels, fill_value, dtype=dtype)
        
        valid_mask = indices < total_voxels
        arr_flat[indices[valid_mask]] = values[valid_mask]
        
        return arr_flat.reshape(shape)
    except Exception as e:
        print(f"❌ Error reading {os.path.basename(csv_path)}: {e}")
        return None

def get_dvh(dose_array, mask_array):
    """ Calculate DVH for binary mask """
    voxels = dose_array[mask_array > 0]
    if len(voxels) == 0:
        return None, None

    voxels = np.sort(voxels)
    num_voxels = len(voxels)
    y_axis = np.arange(num_voxels, 0, -1) / num_voxels * 100
    return voxels, y_axis

# --- CHANGED: Now accepts 'patient_path' as an input ---
def evaluate_patient(patient_path):
    patient_id = os.path.basename(patient_path)
    print(f"\n🏥 Clinical Evaluation for: {patient_id}")
    
    # 1. Load CT & Dose
    ct = load_sparse_volume(os.path.join(patient_path, "ct.csv"), fill_value=-1000, dtype=np.int16)
    true_dose = load_sparse_volume(os.path.join(patient_path, "dose.csv"), fill_value=0, dtype=np.float32)

    # 2. Find the Best Mask
    mask_files = ["PTV70.csv", "PTV56.csv", "possible_dose_mask.csv"]
    mask = None
    mask_name = "Unknown"

    for name in mask_files:
        p = os.path.join(patient_path, name)
        if os.path.exists(p):
            print(f"   ✅ Found structure: {name}")
            mask = load_sparse_volume(p, fill_value=0, dtype=np.uint8)
            mask_name = name
            break
            
    if mask is None:
        print("❌ Critical: No structure CSV found.")
        return

    # 3. AI Prediction
    print("   Running AI Model...")
    model = SimpleUNet3D().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
        
    model.eval()
    
    ct_norm = np.clip(ct, -1000, 1000)
    ct_norm = (ct_norm - -1000) / 2000.0
    inp = torch.from_numpy(ct_norm).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    with torch.no_grad():
        pred = model(inp)
    
    pred_dose = pred.squeeze().cpu().numpy() * 80.0
    
    # 4. Compute DVH
    print("   Calculating DVH Curves...")
    d_real, v_real = get_dvh(true_dose, mask)
    d_pred, v_pred = get_dvh(pred_dose, mask)

    # 5. Plot
    plt.figure(figsize=(10, 7))
    if d_real is not None:
        plt.plot(d_real, v_real, 'k-', linewidth=2.5, label='Clinical Plan (Truth)')
        plt.plot(d_pred, v_pred, 'r--', linewidth=2.5, label='AI Prediction (Ours)')
        
        plt.title(f"DVH Analysis: {mask_name} (Patient {patient_id})")
        plt.xlabel("Dose (Gy)")
        plt.ylabel("Volume (%)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        mean_diff = np.mean(d_pred) - np.mean(d_real)
        
        # D95 Metric
        idx95 = int(len(d_real) * 0.95)
        d95_real = d_real[-idx95] if idx95 < len(d_real) else 0
        d95_pred = d_pred[-idx95] if idx95 < len(d_pred) else 0
        
        stats = (f"Mean Diff: {mean_diff:.2f} Gy\n"
                 f"D95 Diff:  {d95_pred - d95_real:.2f} Gy")
        
        plt.text(0.05, 0.5, stats, transform=plt.gca().transAxes, 
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        print(f"✅ Plot Generated. Mean Dose Diff: {mean_diff:.2f} Gy")
        plt.show()

# --- MAIN BLOCK: PICKS A RANDOM PATIENT ---
if __name__ == "__main__":
    if os.path.exists(VALIDATION_DIR):
        # 1. Get list of all subfolders (patients)
        all_patients = [f for f in os.listdir(VALIDATION_DIR) if os.path.isdir(os.path.join(VALIDATION_DIR, f))]
        
        if len(all_patients) > 0:
            # 2. Pick one randomly
            random_pt = random.choice(all_patients)
            full_path = os.path.join(VALIDATION_DIR, random_pt)
            
            # 3. Run evaluation
            evaluate_patient(full_path)
        else:
            print("❌ No patient folders found in validation directory.")
    else:
        print(f"❌ Directory not found: {VALIDATION_DIR}")