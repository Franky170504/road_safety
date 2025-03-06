from ultralytics import YOLO
import cv2

# Load trained YOLO model
# model = YOLO("best.pt")
model = YOLO("results/helmet_plate/weights/best.pt")

# Load test image
image_path = r"om.jpg"
image = cv2.imread(image_path)

# Perform detection
results = model(image)

# Show results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class ID
        
        # Draw bounding box
        label = model.names[cls]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show image
cv2.imshow("Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
