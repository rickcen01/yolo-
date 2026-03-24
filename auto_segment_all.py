
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
IMAGE_DIR = r"D:\biyesheji\gas"
OUTPUT_VIS_DIR = r"D:\biyesheji\gas_visualized_seg"
LOG_FILE = r"D:\biyesheji\auto_segment_log.txt"
MODEL_NAME = 'yolo11x-seg.pt'
CONF_THRESHOLD = 0.05
SOURCE_CLASS_IDS = [39, 71, 41, 56, 10, 75] 
# 39: bottle
# 71: sink
# 41: cup
# 56: chair
# 10: fire hydrant
# 75: vase

TARGET_CLASS_ID = 0  # gas_cylinder

def log(msg):
    print(msg)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except:
        pass

def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    
    log(f"Loading model {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        log(f"Failed to load model: {e}")
        return

    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    log(f"Found {len(files)} images.")
    
    count = 0
    success_count = 0
    
    for file in files:
        image_path = os.path.join(IMAGE_DIR, file)
        txt_path = os.path.join(IMAGE_DIR, os.path.splitext(file)[0] + ".txt")
        
        try:
            # Run inference
            results = model(image_path, conf=CONF_THRESHOLD, device='cpu', verbose=False)
            result = results[0]
            
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            
            detected_segments = []
            
            if result.masks:
                for i, mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    cls_name = result.names[cls_id]
                    
                    if cls_id in SOURCE_CLASS_IDS:
                        # Get normalized polygon coordinates
                        # result.masks.xyn[i] is a numpy array of shape (N, 2)
                        polygon_norm = result.masks.xyn[i]
                        
                        if len(polygon_norm) < 3:
                            continue
                            
                        # Format: class_id x1 y1 x2 y2 ...
                        segment_line = [str(TARGET_CLASS_ID)]
                        for pt in polygon_norm:
                            segment_line.append(f"{pt[0]:.6f}")
                            segment_line.append(f"{pt[1]:.6f}")
                        
                        detected_segments.append(" ".join(segment_line))
                        
                        # Visualization
                        # Convert normalized to pixel for drawing
                        pts = (polygon_norm * np.array([w, h])).astype(np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                        
                        # Fill specific color for visualization
                        overlay = img.copy()
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
                        
                        # Label
                        label_x, label_y = pts[0][0]
                        cv2.putText(img, f"{cls_name} {conf:.2f}", (int(label_x), int(label_y) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save labels if found
            if detected_segments:
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(detected_segments))
                
                # Save visualization
                vis_path = os.path.join(OUTPUT_VIS_DIR, file)
                cv2.imwrite(vis_path, img)
                
                log(f"[{count+1}/{len(files)}] {file}: Saved {len(detected_segments)} masks.")
                success_count += 1
            else:
                log(f"[{count+1}/{len(files)}] {file}: No relevant objects found.")
                # Maybe clear the txt file if it existed? Or leave it?
                # If we want to strictly overwrite, we should probably clear it.
                # But to be safe, if we find nothing, maybe we shouldn't delete existing valid labels?
                # User said "you to identify its edge... and draw".
                # I'll verify if I should overwrite empty. 
                # For now, I won't overwrite if nothing found, to preserve manual work if any.
        
        except Exception as e:
            log(f"Error processing {file}: {e}")
            
        count += 1

    log(f"Done. Processed {count} images, found segments in {success_count}.")

if __name__ == "__main__":
    main()
