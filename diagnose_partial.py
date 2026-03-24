from ultralytics import YOLO
import os
import sys

# Directory
image_dir = r"d:\biyesheji\gas"
if not os.path.exists(image_dir):
    print(f"Error: Directory {image_dir} does not exist.")
    sys.exit(1)

files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
print(f"Found {len(files)} images in {image_dir}")
print(f"First 3 files: {files[:3]}")

if not files:
    print("No images found.")
    sys.exit(0)

# Load model
print("Loading YOLO11x...")
model = YOLO('yolo11x.pt')

print(f"Diagnostics for partial gas cylinders (Top 3 images)...")

for file in files[:3]:
    img_path = os.path.join(image_dir, file)
    print(f"\nAnalyzing: {file}")
    
    # Use very low confidence to catch anything
    results = model(img_path, conf=0.05, device='cpu', verbose=False)
    
    for result in results:
        if len(result.boxes) == 0:
            print("  No objects detected.")
        else:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id]
                conf = float(box.conf[0])
                print(f"  -> Detected: {name} (ID: {cls_id}) - Conf: {conf:.3f}")

print("Done.")
