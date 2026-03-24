
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Specific problem files mentioned by user
PROBLEM_FILES = [
    "Pic_gas_1.png",   # False positive in bottom left
    "Pic_gas_7.png",   # False positives
    "Pic_gas_12.png",  # False positives
    "Pic_gas_30.png",  # False positives
    "Pic_gas_100.png", # Missed / False positive
    "Pic_gas_124.png", # Missed
    "Pic_gas_183.png"  # Missed
]

IMAGE_DIR = r"D:\biyesheji\gas"
MODEL_NAME = 'yolo11x-seg.pt'

def main():
    print(f"Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    for file in PROBLEM_FILES:
        path = os.path.join(IMAGE_DIR, file)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        print(f"\nAnalyzing {file}...")
        # Use very low threshold to see everything
        results = model(path, conf=0.01, device='cpu', verbose=False)
        result = results[0]
        
        img = cv2.imread(path)
        h, w = img.shape[:2]
        center_x, center_y = w / 2, h / 2
        total_area = h * w
        
        if result.boxes:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate box center and area
                box_w = x2 - x1
                box_h = y2 - y1
                area = box_w * box_h
                area_ratio = area / total_area
                
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                dist_to_center = ((cx - center_x)**2 + (cy - center_y)**2)**0.5
                dist_ratio = dist_to_center / ((w**2 + h**2)**0.5) # Normalize by diagonal
                
                print(f"  ID:{i} | {cls_name} (Class {cls_id}) | Conf: {conf:.3f}")
                print(f"       Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                print(f"       Area Ratio: {area_ratio:.4f} | Dist to Center: {dist_ratio:.4f}")
                
                # Draw all detections for debug
                color = (0, 0, 255) # Red for everything
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            print("  No detections at conf=0.01")
            
        # Save debug image
        debug_path = os.path.join(r"D:\biyesheji", f"debug_{file}")
        cv2.imwrite(debug_path, img)
        print(f"  Saved debug image to {debug_path}")

if __name__ == "__main__":
    main()
