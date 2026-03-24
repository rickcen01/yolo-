
import os
import cv2
import numpy as np
import torch
import sys

print("Starting debug script...", flush=True)

# Check GPU
print("Checking CUDA...", flush=True)
if torch.cuda.is_available():
    print(f"CUDA Available: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("CUDA NOT Available", flush=True)

# Check Input Dir
INPUT_DIR = r"E:\gas"
if os.path.exists(INPUT_DIR):
    files = os.listdir(INPUT_DIR)
    print(f"Found {len(files)} files in {INPUT_DIR}", flush=True)
    
    # Try reading first image
    if len(files) > 0:
        first_file = files[0]
        path = os.path.join(INPUT_DIR, first_file)
        print(f"Reading {path}...", flush=True)
        
        try:
            # Test simple cv2.imread first (might fail with non-ascii chars if any, but E:\gas looks ascii)
            img = cv2.imread(path)
            if img is None:
                print("cv2.imread failed (None)", flush=True)
                # Try the numpy method
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                     print("cv2.imdecode failed (None)", flush=True)
                else:
                     print(f"cv2.imdecode success: {img.shape}", flush=True)
            else:
                print(f"cv2.imread success: {img.shape}", flush=True)
        except Exception as e:
            print(f"Error reading image: {e}", flush=True)
else:
    print(f"{INPUT_DIR} does not exist", flush=True)

print("Debug finished.", flush=True)
