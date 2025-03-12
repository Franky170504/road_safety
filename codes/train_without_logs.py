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

    # Load YOLOv8n model (you can use 'yolov8s.pt' for better accuracy)
    model = YOLO(r"results\helmet_plate\weights\last.pt") 

    # Train the model
    model.train(
        data="data.yaml",     # Path to dataset configuration file
        epochs=250,           # Number of training epochs
        batch=32,             # Batch size (corrected from batch_size)
        imgsz=416,            # Lower resolution for speed
        device="cuda",        # Use GPU (if available)
        workers=2,            # Adjust based on CPU cores
        optimizer="AdamW",    # AdamW is better for YOLOv5
        lr0=0.002,            # Initial learning rate
        lrf=0.01,             # Final learning rate fraction
        momentum=0.937,       # Momentum for SGD (default works well)
        weight_decay=0.0005,  # L2 regularization to avoid overfitting
        dropout=0.1,          # Prevents overfitting
        val=True,             # Enable validation
        cache=False,          # Disable caching (use if you have RAM)
        project="results",    # Store logs and checkpoints (fixed typo in "resuts")
        name="helmet_plate",  # Experiment name
        exist_ok=True,   
        resume=True,     # Overwrite if exists
    )
