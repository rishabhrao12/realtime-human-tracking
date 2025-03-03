import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Motion detection parameters
MOTION_THRESHOLD = 2.0  # Optical Flow magnitude threshold

# Start Webcam Feed
cap = cv2.VideoCapture(0)  # 0 = Default Webcam

# Set resolution (Optional)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize Optical Flow variables
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frames left

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute Optical Flow (Farneback method)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude of flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Update previous frame
    prev_gray = gray.copy()

    # Run YOLO detection
    results = model(frame)
    moving_objects = []  # Store only moving detections

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls) == 0:  # Filter only "person" class (COCO ID = 0)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Compute motion within the bounding box
                motion_region = magnitude[y1:y2, x1:x2]
                motion_score = np.mean(motion_region)

                # Keep only moving objects
                if motion_score > MOTION_THRESHOLD:
                    moving_objects.append((x1, y1, x2, y2))

    # Draw bounding boxes only for moving objects
    for (x1, y1, x2, y2) in moving_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(frame, "Moving", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display output frame
    cv2.imshow("Live YOLO + Optical Flow Motion Filtering", frame)

    # Small delay to reduce CPU usage (adjust if needed)
    if cv2.waitKey(10) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
