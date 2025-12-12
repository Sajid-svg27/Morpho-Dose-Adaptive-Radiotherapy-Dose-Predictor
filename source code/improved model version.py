import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import os
import matplotlib.pyplot as plt

# Import from your existing file
from kbp_model import KBPDataset, SimpleUNet3D

# --- CONFIGURATION ---
DATA_DIR = "../data/processed"
MODEL_LOAD_PATH = "../models/best_model.pth"     # Load the previous best
MODEL_SAVE_DIR = "../models"
BATCH_SIZE = 2        
LEARNING_RATE = 5e-4   # Slightly higher to push the model
EPOCHS = 30            # Add 30 more epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def continue_training():
    print(f"🚀 Resuming Training on {DEVICE}...")
    
    # 1. Prepare Data
    full_dataset = KBPDataset(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Existing Model
    model = SimpleUNet3D().to(DEVICE)
    if os.path.exists(MODEL_LOAD_PATH):
        print(f"   Derived from: {MODEL_LOAD_PATH}")
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    else:
        print("   ⚠️ Previous model not found! Starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() 

    # 3. Training Loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf') # We want to beat the previous best, but let's reset for this run

    start_total = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Train
        for i, (ct, true_dose) in enumerate(train_loader):
            ct, true_dose = ct.to(DEVICE), true_dose.to(DEVICE)
            
            optimizer.zero_grad()
            pred_dose = model(ct)
            loss = criterion(pred_dose, true_dose)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ct, true_dose in val_loader:
                ct = ct.to(DEVICE)
                true_dose = true_dose.to(DEVICE)
                pred = model(ct)
                loss = criterion(pred, true_dose)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        # Save Checkpoint (Overwriting best_model.pth is fine, or create new)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # We overwrite the old best model so the evaluation script picks it up automatically
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))
            print("   Model Improved & Saved! ⭐")

    print(f"\n✅ Extended Training Complete.")

if __name__ == "__main__":
    continue_training()