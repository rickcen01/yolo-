from ultralytics import YOLO
import os

# 1. Load the model
# Using YOLO11n (Nano) for speed on robot arm, or change to 'yolo11s.pt' for better accuracy
model = YOLO('yolo11n.pt')  # pretrained YOLO11n model

# 2. Train the model
# We assume the data.yaml is ready and located at 'd:/biyesheji/gas_dataset/data.yaml'
# You need to label your data first!
if __name__ == '__main__':
    # Train for 100 epochs, image size 640
    results = model.train(
        data='d:/biyesheji/gas_dataset/data.yaml', 
        epochs=100, 
        imgsz=640, 
        device='0', # Use '0' for GPU or 'cpu' for CPU
        project='d:/biyesheji/runs/detect', 
        name='gas_cylinder_v1'
    )

    # 3. Validate
    metrics = model.val()
    print(metrics.box.map)  # map50-95
