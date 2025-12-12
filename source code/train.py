import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import os
import sys


from kbp_model import KBPDataset, SimpleUNet3D


DATA_DIR = "../data/processed"

MODEL_LOAD_PATH = "../models/best_model.pth"     

MODEL_SAVE_PATH = "../models/best_model_overnight.pth"

BATCH_SIZE = 2        
START_LR = 3e-4        
EPOCHS = 50           
DEVICE = "cpu"         

def train_overnight():
    print(f"🌙 Starting OVERNIGHT Training on {DEVICE}...")
    print(f"   Target: {EPOCHS} Epochs")
    print(f"   Strategy: Adam + ReduceLROnPlateau Scheduler")

    
    full_dataset = KBPDataset(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    

    model = SimpleUNet3D().to(DEVICE)
    if os.path.exists(MODEL_LOAD_PATH):
        print(f"    Loading previous brain: {MODEL_LOAD_PATH}")
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    else:
        print("    No previous model found. Starting fresh.")

    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    criterion = nn.MSELoss() 
    
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    
    best_val_loss = float('inf') 
    start_total = time.time()

    try:
        for epoch in range(EPOCHS):
            epoch_start = time.time()
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
            
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            
            epoch_time = time.time() - epoch_start
            remaining_time = epoch_time * (EPOCHS - (epoch + 1))
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                  f"LR: {old_lr:.1e} | "
                  f"Time: {epoch_time:.0f}s (Est. rem: {remaining_time/3600:.1f}h)")

            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"    New Best Saved! ({MODEL_SAVE_PATH})")

    except KeyboardInterrupt:
        print("\n Interrupt received! Saving current state before exiting...")
        torch.save(model.state_dict(), "../models/interrupted_model.pth")
        print("   ✅ Safety save complete. Good morning!")
        sys.exit(0)

    total_time = time.time() - start_total
    print(f"\n✅ Marathon Complete in {total_time/3600:.1f} hours.")

if __name__ == "__main__":
    train_overnight()