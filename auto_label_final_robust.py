
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# Configuration
# ==========================================
IMAGE_DIR = r"D:\biyesheji\gas"
OUTPUT_VIS_DIR = r"D:\biyesheji\gas_visualized_final"
MODEL_NAME = 'yolo11x-seg.pt' 

# Use a very low threshold to ensure we detect faint objects (like in Pic_gas_124)
# We will filter out noise using geometric logic.
CONF_THRESHOLD = 0.01

# Allowed classes and their priority weights
# Higher weight = preferred class if multiple valid objects are found
CLASS_WEIGHTS = {
    39: 2.0,  # bottle (The ideal class)
    71: 1.0,  # sink (Often the valve/top)
    61: 1.0,  # toilet (Seen in Pic_gas_183)
    41: 0.8,  # cup
    56: 0.8,  # chair
    10: 1.0,  # fire hydrant
    75: 0.8,  # vase
}

TARGET_CLASS_ID = 0  # We map everything to 'gas_cylinder'

# Geometric Filters
MIN_AREA_RATIO = 0.03   # Object must be at least 3% of the image
MAX_DIST_RATIO = 0.35   # Object center must be within 35% of the image diagonal from the image center

def main():
    print(f"Initializing Robust Auto-Labeling...", flush=True)
    
    # Clean/Create output directory
    if not os.path.exists(OUTPUT_VIS_DIR):
        os.makedirs(OUTPUT_VIS_DIR)
    
    print(f"Loading model {MODEL_NAME}...", flush=True)
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        return

    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(files)} images.", flush=True)
    
    success_count = 0
    
    for file in files:
        image_path = os.path.join(IMAGE_DIR, file)
        txt_path = os.path.join(IMAGE_DIR, os.path.splitext(file)[0] + ".txt")
        
        try:
            print(f"Processing {file}...", flush=True)
            
            # 1. Inference
            results = model(image_path, conf=CONF_THRESHOLD, device='cpu', verbose=False)
            result = results[0]
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"  Error: Failed to read image.", flush=True)
                continue
                
            h, w = img.shape[:2]
            center_x, center_y = w / 2, h / 2
            total_area = w * h
            diag_length = (w**2 + h**2)**0.5
            
            candidates = []
            
            # 2. Candidate Filtering & Scoring
            if result.masks:
                for i, mask in enumerate(result.masks.data):
                    cls_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    
                    if cls_id in CLASS_WEIGHTS:
                        # Geometric calculations
                        x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                        box_area = (x2 - x1) * (y2 - y1)
                        area_ratio = box_area / total_area
                        
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        dist_to_center = ((cx - center_x)**2 + (cy - center_y)**2)**0.5
                        dist_ratio = dist_to_center / diag_length
                        
                        # --- FILTERING RULES ---
                        # Rule 1: Must be substantial size
                        if area_ratio < MIN_AREA_RATIO:
                            continue
                            
                        # Rule 2: Must be central
                        if dist_ratio > MAX_DIST_RATIO:
                            continue
                            
                        # --- SCORING ---
                        # Score = (Base Confidence) * (Class Priority) * (Centrality Bonus)
                        # Centrality Bonus: 1.0 at edge, up to 2.0 at center
                        centrality_bonus = 1.0 + (1.0 - (dist_ratio / MAX_DIST_RATIO))
                        score = conf * CLASS_WEIGHTS[cls_id] * centrality_bonus
                        
                        candidates.append({
                            'index': i,
                            'cls_id': cls_id,
                            'cls_name': result.names[cls_id],
                            'conf': conf,
                            'score': score,
                            'polygon_norm': result.masks.xyn[i]
                        })
            
            # 3. Selection
            selected = None
            if candidates:
                # Pick the one with the highest custom score
                candidates.sort(key=lambda x: x['score'], reverse=True)
                selected = candidates[0]
            
            # 4. Saving & Visualization
            if selected:
                polygon_norm = selected['polygon_norm']
                
                if len(polygon_norm) >= 3:
                    # Write .txt label
                    segment_line = [str(TARGET_CLASS_ID)]
                    for pt in polygon_norm:
                        segment_line.append(f"{pt[0]:.6f}")
                        segment_line.append(f"{pt[1]:.6f}")
                    
                    with open(txt_path, 'w') as f:
                        f.write(" ".join(segment_line))
                    
                    # Draw Visualization
                    # 1. Draw polygon fill
                    pts = (polygon_norm * np.array([w, h])).astype(np.int32).reshape((-1, 1, 2))
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0)) # Green fill
                    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
                    
                    # 2. Draw thick outline (Edge)
                    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                    
                    # 3. Draw Info Box
                    label = f"{selected['cls_name']} ({selected['conf']:.2f})"
                    label_x, label_y = pts[0][0]
                    cv2.putText(img, label, (int(label_x), int(label_y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    vis_path = os.path.join(OUTPUT_VIS_DIR, file)
                    cv2.imwrite(vis_path, img)
                    print(f"  [SUCCESS] Saved: {label}", flush=True)
                    success_count += 1
                else:
                    print("  [SKIP] Invalid polygon shape.", flush=True)
            else:
                print("  [FAIL] No valid object found.", flush=True)
                # Ensure we don't leave old wrong labels
                with open(txt_path, 'w') as f:
                    pass

        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)

    print(f"\nProcessing Complete.", flush=True)
    print(f"Successfully generated labels for {success_count}/{len(files)} images.", flush=True)
    print(f"Check visualizations in: {OUTPUT_VIS_DIR}", flush=True)

if __name__ == "__main__":
    main()
