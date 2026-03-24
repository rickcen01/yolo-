
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
IMAGE_DIR = r"D:\biyesheji\gas"
OUTPUT_VIS_DIR = r"D:\biyesheji\gas_visualized_seg"
MODEL_NAME = 'yolo11x-seg.pt' 

# Low threshold to catch everything, then we filter by logic
CONF_THRESHOLD = 0.01

# COCO classes relevant to gas cylinder parts (and common misclassifications)
SOURCE_CLASS_IDS = [
    39, # bottle (Correct)
    71, # sink (Valve/Top)
    61, # toilet (Pic_gas_183 misclassification)
    41, # cup
    56, # chair
    10, # fire hydrant
    75, # vase
]

TARGET_CLASS_ID = 0  # gas_cylinder

def main():
    print(f"Starting script...", flush=True)
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    
    print(f"Loading model {MODEL_NAME}...", flush=True)
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        return

    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(files)} images in {IMAGE_DIR}.", flush=True)
    
    count = 0
    success_count = 0
    
    for file in files:
        image_path = os.path.join(IMAGE_DIR, file)
        txt_path = os.path.join(IMAGE_DIR, os.path.splitext(file)[0] + ".txt")
        
        try:
            print(f"Processing {file}...", flush=True)
            
            # Run inference
            results = model(image_path, conf=CONF_THRESHOLD, device='cpu', verbose=False)
            result = results[0]
            
            img = cv2.imread(image_path)
            if img is None: continue
            h, w = img.shape[:2]
            center_x, center_y = w / 2, h / 2
            total_area = w * h
            
            candidates = []
            
            if result.masks:
                for i, mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    
                    if cls_id in SOURCE_CLASS_IDS:
                        # Get bbox to calculate metrics
                        x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                        box_area = (x2 - x1) * (y2 - y1)
                        area_ratio = box_area / total_area
                        
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        dist_to_center = ((cx - center_x)**2 + (cy - center_y)**2)**0.5
                        # Normalize distance by diagonal
                        diag = (w**2 + h**2)**0.5
                        dist_ratio = dist_to_center / diag
                        
                        # FILTER 1: Area Threshold (5%)
                        if area_ratio < 0.05:
                            continue
                            
                        # FILTER 2: Centrality (35% from center)
                        if dist_ratio > 0.35:
                            continue
                            
                        candidates.append({
                            'index': i,
                            'cls_id': cls_id,
                            'conf': conf,
                            'area_ratio': area_ratio,
                            'dist_ratio': dist_ratio,
                            'polygon_norm': result.masks.xyn[i]
                        })
            
            # SELECTION LOGIC
            selected_candidate = None
            if candidates:
                # Sort by distance to center (ascending)
                candidates.sort(key=lambda x: x['dist_ratio'])
                
                # Pick the most central one
                selected_candidate = candidates[0]
            
            # Save Logic
            if selected_candidate:
                polygon_norm = selected_candidate['polygon_norm']
                
                if len(polygon_norm) >= 3:
                    # Save Label
                    segment_line = [str(TARGET_CLASS_ID)]
                    for pt in polygon_norm:
                        segment_line.append(f"{pt[0]:.6f}")
                        segment_line.append(f"{pt[1]:.6f}")
                    
                    with open(txt_path, 'w') as f:
                        f.write(" ".join(segment_line))
                    
                    # Visualization
                    pts = (polygon_norm * np.array([w, h])).astype(np.int32).reshape((-1, 1, 2))
                    
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
                    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                    
                    cls_name = result.names[selected_candidate['cls_id']]
                    conf = selected_candidate['conf']
                    label_x, label_y = pts[0][0]
                    text = f"{cls_name} {conf:.2f} (Dist:{selected_candidate['dist_ratio']:.2f})"
                    cv2.putText(img, text, (int(label_x), int(label_y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    vis_path = os.path.join(OUTPUT_VIS_DIR, file)
                    cv2.imwrite(vis_path, img)
                    print(f"  Saved match: {cls_name} (Conf {conf:.2f}, Dist {selected_candidate['dist_ratio']:.2f})", flush=True)
                    success_count += 1
                else:
                    print("  Invalid polygon (points < 3)", flush=True)
            else:
                print("  No valid objects found after filtering.", flush=True)
                with open(txt_path, 'w') as f:
                    pass # Empty file

        except Exception as e:
            print(f"Error processing {file}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            
        count += 1

    print(f"Done. Processed {count} images, successfully segmented {success_count}.", flush=True)

if __name__ == "__main__":
    main()
