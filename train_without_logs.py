from ultralytics import YOLO
import os
import shutil


# Function to delete old training logs to save space
def clean_old_logs():
    log_dirs = ["runs/train"]
    for log_dir in log_dirs:
        try:
            shutil.rmtree(log_dir)  # Deletes the directory
            print(f"Deleted old logs at {log_dir}")
        except FileNotFoundError:
            pass  # Ignore if directory doesn't exist

if __name__ == '__main__':
    # Clean logs before training
    clean_old_logs()

    # Load YOLOv8n model
    model = YOLO("results/helmet_plate/weights/last.pt")

    # Train the model with optimized hyperparameters
    model.train(
        data="data.yaml",       # Path to dataset config file
        epochs=300,             # More epochs for better convergence
        batch=12,               # Optimized batch size for RTX 4050 (6GB VRAM)
        imgsz=640,              # Standard YOLO image size
        device=0,               # Use GPU
        workers=4,              # Set workers based on CPU cores
        optimizer="AdamW",      # AdamW optimizer for better convergence
        lr0=0.0018,             # Adjusted initial learning rate
        lrf=0.2,                # Increased final learning rate fraction
        momentum=0.95,          # Higher momentum for better optimization
        weight_decay=0.0004,    # Lower weight decay to reduce overfitting
        dropout=0.2,            # Higher dropout for better regularization
        val=True,               # Enable validation during training
        cache=True,             # Use caching for faster training
        amp=True,               # Enable mixed precision training (efficient training)
        hsv_h=0.015,            # Hue augmentation
        hsv_s=0.7,              # Saturation augmentation
        hsv_v=0.4,              # Value augmentation
        mosaic=1.0,             # Enable mosaic augmentation
        mixup=0.2,              # Enable mixup for better generalization
        project="results",      # Store logs and weights
        name="helmet_plate",    # Experiment name
        exist_ok=True,    
        resume=True,    # Overwrite existing runs
    )
