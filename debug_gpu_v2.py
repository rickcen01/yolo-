
import os
import sys

print("STEP 1: Started", flush=True)

try:
    import torch
    print("STEP 2: Torch imported", flush=True)
except Exception as e:
    print(f"STEP 2 FAIL: {e}", flush=True)

try:
    print("STEP 3: Checking CUDA...", flush=True)
    if torch.cuda.is_available():
        print(f"STEP 3: CUDA Available: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("STEP 3: CUDA NOT Available", flush=True)
except Exception as e:
    print(f"STEP 3 FAIL: {e}", flush=True)

INPUT_DIR = r"E:\gas"
print(f"STEP 4: Checking dir {INPUT_DIR}", flush=True)

if os.path.exists(INPUT_DIR):
    try:
        files = os.listdir(INPUT_DIR)
        print(f"STEP 5: Found {len(files)} files", flush=True)
    except Exception as e:
        print(f"STEP 5 FAIL: {e}", flush=True)
else:
    print("STEP 4: Dir not found", flush=True)

print("STEP 6: Done", flush=True)
