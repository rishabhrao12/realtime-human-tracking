import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Motion detection threshold
MOTION_THRESHOLD = 2.0  # Controls how much movement is needed to track an object

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = Default webcam

# Set resolution (Optional)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize Optical Flow variables
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Initialize Kalman Filter for tracking
kf = cv2.KalmanFilter(4, 2)
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
kf.errorCovPost = np.eye(4, dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
    moving_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls) == 0:  # Filter only "person" class
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Compute motion inside bounding box
                motion_region = magnitude[y1:y2, x1:x2]
                motion_score = np.mean(motion_region)

                # Keep only moving objects
                if motion_score > MOTION_THRESHOLD:
                    # Kalman Filter prediction
                    predicted = kf.predict()
                    predicted_x, predicted_y = int(predicted[0]), int(predicted[1])

                    # Kalman Filter correction using YOLO detection
                    measurement = np.array([[np.float32((x1 + x2) / 2)], [np.float32((y1 + y2) / 2)]])
                    corrected = kf.correct(measurement)

                    # Store the corrected position
                    corrected_x, corrected_y = int(corrected[0]), int(corrected[1])
                    moving_objects.append((x1, y1, x2, y2, corrected_x, corrected_y))

    # Draw bounding boxes for moving objects
    for (x1, y1, x2, y2, corrected_x, corrected_y) in moving_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Moving", (corrected_x, corrected_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display output frame
    cv2.imshow("YOLO + Optical Flow + Kalman Filter Tracking", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
