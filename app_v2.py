
import os
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil

# ==========================================
# Segmentation Engine
# ==========================================
class GasCylinderSegmenter:
    def __init__(self, model_path='yolo11x-seg.pt'):
        print(f"Loading model {model_path}...", flush=True)
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
        
        # Inference
        results = self.model(img, conf=self.conf_threshold, device='cpu', verbose=False)
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

# ==========================================
# FastAPI App
# ==========================================

app = FastAPI()

# Mount static files
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Segmenter
segmenter = GasCylinderSegmenter(model_path='yolo11x-seg.pt')

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image file"}

    # Process image
    processed_img, segment_data = segmenter.process_image(img)
    
    # Save processed image to a temp file
    output_filename = f"processed_{file.filename}"
    output_path = os.path.join("static", output_filename)
    cv2.imwrite(output_path, processed_img)
    
    return {
        "image_url": f"/static/{output_filename}",
        "segment_data": segment_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
