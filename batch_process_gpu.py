
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys

# ==========================================
# Segmentation Engine (Adapted from app_v2.py)
# ==========================================
class GasCylinderSegmenter:
    def __init__(self, model_path='yolo11x-seg.pt', device='0'):
        print(f"Loading model {model_path} on device {device}...", flush=True)
        self.device = device
        self.model = YOLO(model_path)
        
        # Configuration
        self.conf_threshold = 0.01
        self.min_area_ratio = 0.03
        self.max_dist_ratio = 0.35
        self.target_class_id = 0
        
        # Class Weights
        self.class_weights = {
            39: 2.0,  # bottle
            71: 1.0,  # sink
            61: 1.0,  # toilet
            41: 0.8,  # cup
            56: 0.8,  # chair
            10: 1.0,  # fire hydrant
            75: 0.8,  # vase
        }

    def process_image(self, img):
        h, w = img.shape[:2]
        center_x, center_y = w / 2, h / 2
        total_area = w * h
        diag_length = (w**2 + h**2)**0.5
        
        # Inference using configured device
        # Note: Ultralytics handles device='0' for GPU 0, or 'cpu'
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
                    
                    if area_ratio < self.min_area_ratio:
                        continue
                    if dist_ratio > self.max_dist_ratio:
                        continue
                        
                    centrality_bonus = 1.0 + (1.0 - (dist_ratio / self.max_dist_ratio))
                    score = conf * self.class_weights[cls_id] * centrality_bonus
                    
                    candidates.append({
                        'index': i,
                        'cls_id': cls_id,
                        'cls_name': result.names[cls_id],
                        'conf': conf,
                        'score': score,
                        'polygon_norm': result.masks.xyn[i]
                    })
        
        # Selection
        selected = None
        if candidates:
            candidates.sort(key=lambda x: x['score'], reverse=True)
            selected = candidates[0]
            
        processed_img = img.copy()
        segment_data = None
        
        if selected:
            polygon_norm = selected['polygon_norm']
            if len(polygon_norm) >= 3:
                # Prepare return data
                segment_data = {
                    'class_id': self.target_class_id,
                    'polygon': polygon_norm.tolist(), # normalized
                    'label': f"{selected['cls_name']} ({selected['conf']:.2f})"
                }
                
                # Draw Visualization
                pts = (polygon_norm * np.array([w, h])).astype(np.int32).reshape((-1, 1, 2))
                
                # Green fill
                overlay = processed_img.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.4, processed_img, 0.6, 0, processed_img)
                
                # Thick outline
                cv2.polylines(processed_img, [pts], True, (0, 255, 0), 2)
                
                # Label
                label = f"{segment_data['label']}"
                label_x, label_y = pts[0][0]
                cv2.putText(processed_img, label, (int(label_x), int(label_y) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
        return processed_img, segment_data

def main():
    # Configuration
    INPUT_DIR = r"E:\gas"
    OUTPUT_DIR = r"F:\guoji"
    MODEL_PATH = 'yolo11x-seg.pt'
    
    # Check for GPU
    if torch.cuda.is_available():
        device = '0'
        print("CUDA is available. Using GPU.", flush=True)
    else:
        device = 'cpu'
        print("CUDA not found. Using CPU.", flush=True)

    # Verify input directory
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.", flush=True)
        return

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory {OUTPUT_DIR}", flush=True)

    # Initialize Segmenter
    try:
        segmenter = GasCylinderSegmenter(model_path=MODEL_PATH, device=device)
    except Exception as e:
        print(f"Error initializing segmenter: {e}", flush=True)
        return

    # Process images
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Found {len(files)} images to process.", flush=True)

    success_count = 0
    
    for i, file in enumerate(files):
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file)
        
        try:
            print(f"[{i+1}/{len(files)}] Processing {file}...", flush=True)
            
            # Read image
            # Handle unicode paths properly using numpy
            img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"  Error: Failed to read image {file}", flush=True)
                continue

            # Process
            processed_img, data = segmenter.process_image(img)
            
            # Save
            # Handle unicode paths properly using numpy
            is_success, im_buf = cv2.imencode(os.path.splitext(file)[1], processed_img)
            if is_success:
                im_buf.tofile(output_path)
                if data:
                    print(f"  Saved result: {data['label']}", flush=True)
                    success_count += 1
                else:
                    print(f"  Saved result (No object detected)", flush=True)
            else:
                print(f"  Error: Failed to save image {file}", flush=True)
                
        except Exception as e:
            print(f"  Error processing {file}: {e}", flush=True)

    print(f"\nBatch processing complete.", flush=True)
    print(f"Successfully processed {len(files)} files.", flush=True)
    print(f"Found objects in {success_count} files.", flush=True)
    print(f"Results saved to {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
