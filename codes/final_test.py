import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr
import re
import time

# Load YOLO model
model = YOLO("best.pt")

# Initialize EasyOCR with multiple languages and allow lists
ocr = easyocr.Reader(['en'], gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)

# Create directory for saving violations
os.makedirs("violations", exist_ok=True)

# Class IDs
HELMET_CLASS_ID = 0
NO_HELMET_CLASS_ID = 1
NUMBER_PLATE_CLASS_ID = 2

# DeepSORT Tracker
tracker = DeepSort(max_age=30, n_init=5, nms_max_overlap=1.0)

# Dictionary to track recorded violations
recorded_violations = {}
no_helmet_counter = {}
violation_cooldown = {}  # Track time since last violation
COOLDOWN_PERIOD = 60  # Cooldown in seconds before recording same plate again

# Regex pattern for common license plate formats
LICENSE_PLATE_PATTERN = re.compile(r'[A-Z0-9]{4,10}')

# OCR confidence threshold
OCR_CONFIDENCE_THRESHOLD = 0.4

# Open video file
video_path = r'videos/om-7march.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Video file not found or cannot be opened: {video_path}")
    exit()

# Get video metrics for logging
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, Total Frames: {frame_count}")

# Frame counter
frame_counter = 0
last_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    
    # Process only every 2nd frame for performance
    # if frame_counter % 2 != 0:
    #     continue
    
    # Run YOLO detection
    results = model(frame)

    detections = []
    number_plate_list = []

    # Process detections
    for result in results:
        if not result.boxes:
            continue
            
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # Skip low confidence detections
            if conf < 0.4:
                continue

            color = (0, 255, 0) if cls == HELMET_CLASS_ID else (0, 0, 255) if cls == NO_HELMET_CLASS_ID else (255, 0, 0)
            label = "Helmet" if cls == HELMET_CLASS_ID else "No Helmet" if cls == NO_HELMET_CLASS_ID else "Number Plate"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if cls == NO_HELMET_CLASS_ID:
                detections.append(([x1, y1, x2, y2], conf, cls))
            elif cls == NUMBER_PLATE_CLASS_ID:
                number_plate_list.append((x1, y1, x2, y2))

    # Update tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # Current time for cooldown checking
    current_time = time.time()
    
    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltwh = track.to_ltwh()
        x1, y1 = int(ltwh[0]), int(ltwh[1])
        x2, y2 = int(ltwh[0] + ltwh[2]), int(ltwh[1] + ltwh[3])

        # Count frames of No Helmet detection
        if track_id not in no_helmet_counter:
            no_helmet_counter[track_id] = 0
        no_helmet_counter[track_id] += 1

        # Ensure violation is only recorded after consistently detecting no helmet
        if no_helmet_counter[track_id] < 3:  # Increased threshold for stability
            continue

        # Find the closest number plate using improved proximity logic
        closest_plate = None
        min_distance = float("inf")
        
        for plate_box in number_plate_list:
            p_x1, p_y1, p_x2, p_y2 = plate_box
            
            # Calculate center points
            rider_center_x = (x1 + x2) / 2
            rider_center_y = (y1 + y2) / 2
            plate_center_x = (p_x1 + p_x2) / 2
            plate_center_y = (p_y1 + p_y2) / 2
            
            # Calculate horizontal and vertical distances
            horizontal_distance = abs(rider_center_x - plate_center_x)
            vertical_distance = abs(rider_center_y - plate_center_y)
            
            # Only consider plates that are reasonably close horizontally and below the rider
            if horizontal_distance < ltwh[2] * 1.2 and p_y1 > y1 and p_y1 < y2 + ltwh[3] * 1.5:
                distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_plate = plate_box

        if closest_plate:
            p_x1, p_y1, p_x2, p_y2 = closest_plate
            
            # Enhance plate detection and OCR
            plate_crop = frame[p_y1:p_y2, p_x1:p_x2]
            if plate_crop.size == 0:
                continue
                
            # Apply preprocessing pipeline for better OCR
            # Convert to grayscale
            plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            
            # Resize for better OCR (larger)
            plate_gray = cv2.resize(plate_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
            # Apply bilateral filter to remove noise while preserving edges
            plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
            
            # Apply adaptive thresholding
            plate_gray = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 11, 2)
            
            # Morphological operations to clean up the image
            kernel = np.ones((3, 3), np.uint8)
            plate_gray = cv2.morphologyEx(plate_gray, cv2.MORPH_OPEN, kernel)
            plate_gray = cv2.morphologyEx(plate_gray, cv2.MORPH_CLOSE, kernel)
            
            # OCR with additional parameters
            result = ocr.readtext(plate_gray, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            # Process OCR results
            plate_text = "Unknown"
            max_confidence = 0.0
            
            for detection in result:
                text = detection[1]
                confidence = detection[2]
                
                # Skip results with low confidence
                if confidence < OCR_CONFIDENCE_THRESHOLD:
                    continue
                    
                # Clean up the text (remove spaces, special characters)
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                # Check if it matches license plate pattern and has higher confidence
                if LICENSE_PLATE_PATTERN.match(cleaned_text) and confidence > max_confidence:
                    plate_text = cleaned_text
                    max_confidence = confidence
            
            # Add visual feedback for OCR result
            cv2.putText(frame, f"Plate: {plate_text} ({max_confidence:.2f})", 
                      (p_x1, p_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Check cooldown to avoid duplicate recordings
            plate_key = plate_text if plate_text != "Unknown" else f"unknown_{track_id}"
            
            if plate_key in violation_cooldown:
                time_since_last = current_time - violation_cooldown[plate_key]
                if time_since_last < COOLDOWN_PERIOD:
                    continue  # Skip recording if in cooldown period
            
            # Save violation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            violation_folder = f"violations/{timestamp}_{plate_key}"
            os.makedirs(violation_folder, exist_ok=True)
            
            # Get a larger crop around both rider and plate
            padding_x, padding_y = 100, 130
            crop_x1 = max(0, min(x1, p_x1) - padding_x)
            crop_y1 = max(0, min(y1, p_y1) - padding_y)
            crop_x2 = min(frame.shape[1], max(x2, p_x2) + padding_x)
            crop_y2 = min(frame.shape[0], max(y2, p_y2) + padding_y)
            
            # Save full violation image
            violation_img = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            cv2.imwrite(os.path.join(violation_folder, "violation.jpg"), violation_img)
            
            # Save plate crop separately
            cv2.imwrite(os.path.join(violation_folder, "plate.jpg"), plate_crop)
            cv2.imwrite(os.path.join(violation_folder, "plate_processed.jpg"), plate_gray)
            
            # Save metadata
            with open(os.path.join(violation_folder, "metadata.txt"), "w") as f:
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"License Plate: {plate_text}\n")
                f.write(f"OCR Confidence: {max_confidence:.2f}\n")
                f.write(f"Frame: {frame_counter} / {frame_count}\n")
                f.write(f"Video Source: {video_path}\n")
            
            print(f"Violation saved: {violation_folder} | Plate: {plate_text} | Conf: {max_confidence:.2f}")
            
            # Update cooldown timer
            violation_cooldown[plate_key] = current_time
            recorded_violations[track_id] = plate_text

    # Calculate and display FPS
    if frame_counter % 30 == 0:
        current_time = time.time()
        fps_current = 30 / (current_time - last_time)
        last_time = current_time
        cv2.putText(frame, f"FPS: {fps_current:.1f}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Helmet Detection + Number Plate OCR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()