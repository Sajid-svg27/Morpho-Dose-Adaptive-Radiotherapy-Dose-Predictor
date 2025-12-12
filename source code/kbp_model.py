import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os


class KBPDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to the folder containing .npz files
        """
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        
        
        self.ct_min, self.ct_max = -1000.0, 1000.0
        
        self.dose_max = 80.0 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path)
        
        ct = data['ct'].astype(np.float32)
        dose = data['dose'].astype(np.float32)

        
        ct = np.clip(ct, self.ct_min, self.ct_max)
        ct = (ct - self.ct_min) / (self.ct_max - self.ct_min)

        dose = dose / self.dose_max

    
        ct_tensor = torch.from_numpy(ct).unsqueeze(0)
        dose_tensor = torch.from_numpy(dose).unsqueeze(0)

        return ct_tensor, dose_tensor

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SimpleUNet3D(nn.Module):
    """
    A lightweight 3D U-Net.
    Input: CT Volume (1 channel)
    Output: Dose Volume (1 channel)
    """
    def __init__(self):
        super(SimpleUNet3D, self).__init__()

        self.dconv_down1 = DoubleConv(1, 16)
        self.dconv_down2 = DoubleConv(16, 32)
        self.dconv_down3 = DoubleConv(32, 64)

        self.maxpool = nn.MaxPool3d(2)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)        
        
        self.dconv_up2 = DoubleConv(32 + 64, 32)
        self.dconv_up1 = DoubleConv(16 + 32, 16)
        
        self.conv_last = nn.Conv3d(16, 1, kernel_size=1)
        self.relu = nn.ReLU() 

    def forward(self, x):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        
        x = self.upsample(conv3)
        x = torch.cat([x, conv2], dim=1) 
        
        x = self.dconv_up2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1) 
        
        x = self.dconv_up1(x)
        

        out = self.conv_last(x)
        return self.relu(out) 

if __name__ == "__main__":
    
    print("Testing Model Components...")
    
    ds = KBPDataset("../data/processed")
    if len(ds) > 0:
        sample_ct, sample_dose = ds[0]
        print(f" Dataset Loaded. Found {len(ds)} patients.")
        print(f"   Sample CT Shape: {sample_ct.shape} | Range: {sample_ct.min():.2f} - {sample_ct.max():.2f}")
    else:
        print("⚠️ Warning: No .npz files found in ../data/processed")


    model = SimpleUNet3D()
    print(" Model Architecture Created.")
    

    fake_input = torch.randn(1, 1, 128, 128, 128)
    print("   Running dry run (this might take 10 seconds)...")
    
    with torch.no_grad():
        output = model(fake_input)
    
    print(f"   Output Shape: {output.shape}")
    if output.shape == (1, 1, 128, 128, 128):
        print(" SUCCESS! The Brain is ready.")
    else:
        print(" ERROR: Output shape mismatch.")