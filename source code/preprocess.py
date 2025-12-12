import os
import numpy as np
import pandas as pd
import time


RAW_DATA_ROOT = r"C:\Users\sajid\OneDrive\Desktop\KBP Dataset\archive\provided-data" 
OUTPUT_DIR = "../data/processed"

IMG_SHAPE = (128, 128, 128)
TOTAL_VOXELS = 128 * 128 * 128

def reconstruct_volume(csv_path, fill_value=0, dtype=np.float32):
    """ Reconstructs 3D volume from sparse CSV. """
    if not os.path.exists(csv_path):
        return None
        
    try:
        df = pd.read_csv(csv_path)
        indices = df.iloc[:, 0].values.astype(int)
        values = df.iloc[:, 1].values.astype(dtype)
        
        vol_flat = np.full(TOTAL_VOXELS, fill_value, dtype=dtype)
        vol_flat[indices] = values
        return vol_flat.reshape(IMG_SHAPE)
    except Exception as e:
        print(f"     Error reading {os.path.basename(csv_path)}: {e}")
        return None

def process_dataset(subset_name):
    """ Process 'train-pats' or 'validation-pats' """
    subset_path = os.path.join(RAW_DATA_ROOT, subset_name)
    if not os.path.exists(subset_path):
        print(f"ℹ Skipping {subset_name} (Folder not found)")
        return

    patient_list = sorted(os.listdir(subset_path))
    print(f"\n Processing {subset_name}: {len(patient_list)} patients found.")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    count = 0
    start_time = time.time()

    for pat_id in patient_list:
        pat_folder = os.path.join(subset_path, pat_id)
        
        
        ct_path = os.path.join(pat_folder, "ct.csv")
        ct_vol = reconstruct_volume(ct_path, fill_value=-1000, dtype=np.int16)
        
        
        dose_path = os.path.join(pat_folder, "dose.csv")
        dose_vol = reconstruct_volume(dose_path, fill_value=0, dtype=np.float32)

        
        if ct_vol is not None and dose_vol is not None:
            save_name = os.path.join(OUTPUT_DIR, f"{pat_id}.npz")
            np.savez_compressed(save_name, ct=ct_vol, dose=dose_vol)
            count += 1
            
            
            if count % 10 == 0:
                print(f"   Processed {count}/{len(patient_list)} patients...")
        else:
            print(f"    Skipping {pat_id} due to missing/bad data.")

    elapsed = time.time() - start_time
    print(f" Finished {subset_name}. Saved {count} files in {elapsed:.1f}s.")

if __name__ == "__main__":
    print("Starting Batch Processing...")
    process_dataset("train-pats")
    process_dataset("validation-pats")
    print("\n All Done! Data is ready for AI.")