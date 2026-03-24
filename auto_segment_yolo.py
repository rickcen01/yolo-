
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
IMAGE_DIR = r"D:\biyesheji\gas"
OUTPUT_VIS_DIR = r"D:\biyesheji\gas_visualized_seg"
MODEL_NAME = 'yolo11x-seg.pt'  # Strongest model
CONF_THRESHOLD = 0.1
# Lower threshold for difficult images
LOW_CONF_THRESHOLD = 0.01

# COCO classes relevant to gas cylinder parts
SOURCE_CLASS_IDS = [39, 71, 41, 56, 10, 75] 
# 39: bottle
# 71: sink
# 41: cup
# 56: chair
# 10: fire hydrant
# 75: vase

TARGET_CLASS_ID = 0  # gas_cylinder

def main():
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    
    print(f"Loading model {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(files)} images in {IMAGE_DIR}.")
    
    count = 0
    success_count = 0
    
    for file in files:
        image_path = os.path.join(IMAGE_DIR, file)
        txt_path = os.path.join(IMAGE_DIR, os.path.splitext(file)[0] + ".txt")
        
        try:
            print(f"Processing {file}...")
            
            # First pass with normal threshold
            results = model(image_path, conf=CONF_THRESHOLD, device='cpu', verbose=False)
            result = results[0]
            
            # If nothing found, retry with lower threshold
            if not any(int(box.cls[0]) in SOURCE_CLASS_IDS for box in result.boxes):
                print(f"  -> Nothing found, retrying with low confidence {LOW_CONF_THRESHOLD}...")
                results = model(image_path, conf=LOW_CONF_THRESHOLD, device='cpu', verbose=False)
                result = results[0]
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"  Error: Could not read image {file}")
                continue
                
            h, w = img.shape[:2]
            
            detected_segments = []
            valid_masks_found = False
            
            if result.masks:
                for i, mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    cls_name = result.names[cls_id]
                    
                    if cls_id in SOURCE_CLASS_IDS:
                        if result.masks.xyn[i].shape[0] < 3:
                            continue
                            
                        polygon_norm = result.masks.xyn[i]
                        
                        # Format: class_id x1 y1 x2 y2 ...
                        segment_line = [str(TARGET_CLASS_ID)]
                        for pt in polygon_norm:
                            segment_line.append(f"{pt[0]:.6f}")
                            segment_line.append(f"{pt[1]:.6f}")
                        
                        detected_segments.append(" ".join(segment_line))
                        
                        valid_masks_found = True
                        print(f"  Found {cls_name} ({conf:.2f})")
                        
                        # Visualization
                        pts = (polygon_norm * np.array([w, h])).astype(np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                        
                        # Label
                        label_x, label_y = pts[0][0]
                        cv2.putText(img, f"{cls_name} {conf:.2f}", (int(label_x), int(label_y) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save labels if found
            if valid_masks_found:
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(detected_segments))
                
                # Save visualization
                vis_path = os.path.join(OUTPUT_VIS_DIR, file)
                cv2.imwrite(vis_path, img)
                print(f"  Saved labels and visualization.")
                success_count += 1
            else:
                print(f"  No relevant objects found even with low threshold.")
        
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
        count += 1

    print(f"Done. Processed {count} images, found segments in {success_count}.")

if __name__ == "__main__":
    main()
