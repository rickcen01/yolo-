
import os
import cv2
import numpy as np
import sys
import time

print("STEP 0: Imports starting...", flush=True)
try:
    import torch
    print(f"STEP 0: Torch imported ({torch.__version__})", flush=True)
    from ultralytics import YOLO
    print("STEP 0: YOLO imported", flush=True)
except Exception as e:
    print(f"STEP 0 ERROR: {e}", flush=True)
    sys.exit(1)

# ==========================================
# Segmentation Engine
# ==========================================
class GasCylinderSegmenter:
    def __init__(self, model_path='yolo11x-seg.pt', device='0'):
        print(f"  [Segmenter] Loading model {model_path} on device {device}...", flush=True)
        self.device = device
        self.model = YOLO(model_path)
        print(f"  [Segmenter] Model loaded.", flush=True)
        
        # Configuration
        self.conf_threshold = 0.01
        self.min_area_ratio = 0.03
        self.max_dist_ratio = 0.35
        self.target_class_id = 0
        
        self.class_weights = {
            39: 2.0, 71: 1.0, 61: 1.0, 41: 0.8, 56: 0.8, 10: 1.0, 75: 0.8
        }

    def process_image(self, img):
        h, w = img.shape[:2]
        center_x, center_y = w / 2, h / 2
        total_area = w * h
        diag_length = (w**2 + h**2)**0.5
        
        results = self.model(img, conf=self.conf_threshold, device=self.device, verbose=False)
        result = results[0]
        
        candidates = []
        if result.masks:
            for i, mask in enumerate(result.masks.data):
                cls_id = int(result.boxes.cls[i])
                conf = float(result.boxes.conf[i])
                
                if cls_id in self.class_weights:
                    x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / total_area
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    dist_to_center = ((cx - center_x)**2 + (cy - center_y)**2)**0.5
                    dist_ratio = dist_to_center / diag_length
                    
                    if area_ratio < self.min_area_ratio: continue
                    if dist_ratio > self.max_dist_ratio: continue
                        
                    centrality_bonus = 1.0 + (1.0 - (dist_ratio / self.max_dist_ratio))
                    score = conf * self.class_weights[cls_id] * centrality_bonus
                    
                    candidates.append({
                        'index': i, 'cls_id': cls_id, 'cls_name': result.names[cls_id],
                        'conf': conf, 'score': score, 'polygon_norm': result.masks.xyn[i]
                    })
        
        selected = None
        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            selected = candidates[0]
            
        processed_img = img.copy()
        segment_data = None
        
        if selected:
            polygon_norm = selected['polygon_norm']
            if len(polygon_norm) >= 3:
                segment_data = {
                    'class_id': self.target_class_id,
                    'polygon': polygon_norm.tolist(),
                    'label': f"{selected['cls_name']} ({selected['conf']:.2f})"
                }
                pts = (polygon_norm * np.array([w, h])).astype(np.int32).reshape((-1, 1, 2))
                overlay = processed_img.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.4, processed_img, 0.6, 0, processed_img)
                cv2.polylines(processed_img, [pts], True, (0, 255, 0), 2)
                label = f"{segment_data['label']}"
                label_x, label_y = pts[0][0]
                cv2.putText(processed_img, label, (int(label_x), int(label_y) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
        return processed_img, segment_data

def main():
    print("STEP 1: Starting Main", flush=True)
    INPUT_DIR = r"E:\gas"
    OUTPUT_DIR = r"F:\guoji"
    MODEL_PATH = 'yolo11x-seg.pt'
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"STEP 2: Using device: {device}", flush=True)

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.", flush=True)
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"STEP 3: Created output directory {OUTPUT_DIR}", flush=True)

    try:
        segmenter = GasCylinderSegmenter(model_path=MODEL_PATH, device=device)
    except Exception as e:
        print(f"STEP 4 ERROR: Failed to init segmenter: {e}", flush=True)
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"STEP 5: Found {len(files)} images.", flush=True)

    success_count = 0
    
    for i, file in enumerate(files):
        # Optional: Limit for testing if needed, but user wants all
        # if i >= 5: break 
        
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file)
        
        try:
            print(f"[{i+1}/{len(files)}] Reading {file}...", flush=True)
            # Use numpy for unicode support
            img_data = np.fromfile(input_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"  Error: Failed to read image {file}", flush=True)
                continue

            processed_img, data = segmenter.process_image(img)
            
            # Save using imencode for unicode support
            is_success, im_buf = cv2.imencode(os.path.splitext(file)[1], processed_img)
            if is_success:
                im_buf.tofile(output_path)
                if data:
                    print(f"  Saved: {data['label']}", flush=True)
                    success_count += 1
                else:
                    print(f"  Saved (No detection)", flush=True)
            else:
                print(f"  Error: Failed to encode image", flush=True)
                
        except Exception as e:
            print(f"  CRASH processing {file}: {e}", flush=True)

    print(f"\nDONE. Processed {len(files)} files. Success: {success_count}", flush=True)

if __name__ == "__main__":
    main()
