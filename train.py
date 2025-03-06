from ultralytics import YOLO
import os


if __name__ == '__main__':
    # Load YOLOv8n model (nano version, you can try 'yolov8s' for better accuracy)
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="data.yaml",  # Path to dataset config file
        epochs=100,           # Number of training epochs
        batch=16,             # Adjust based on GPU VRAM (Lower if OOM error)
        imgsz=640,            # Input image size (640 is standard)
        device=0,             # Use GPU
        workers=4,            # Adjust based on CPU cores (More workers = faster loading)
        optimizer="AdamW",    # AdamW is better for YOLOv8
        lr0=0.002,            # Initial learning rate
        lrf=0.01,             # Final learning rate fraction
        momentum=0.937,       # Momentum for SGD (Default works well)
        weight_decay=0.0005,  # L2 regularization to avoid overfitting
        dropout=0.1,          # Prevents overfitting
        val=True,             # Enable validation
        cache=True,           # Caches dataset in memory for faster training
        project="D:/YOLO_Training",  # Store logs and checkpoints in D: drive
        name="helmet_plate",   # Experiment name
        exist_ok=True,        # Overwrite if exists
    )
  