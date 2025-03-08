import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model (Replace 'best.pt' with your model file)
model = YOLO("results/helmet_plate/weights/best.pt")  

# Open the webcam (0 for the default webcam, change to 1 if using another camera)
cap = cv2.VideoCapture(0)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Run YOLOv8 on the frame
    results = model(frame)

    # Plot the detections on the frame
    for result in results:
        frame = result.plot()  # Draw the bounding boxes on the frame

    # Display the frame
    cv2.imshow("Helmet Detection - YOLOv8", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()