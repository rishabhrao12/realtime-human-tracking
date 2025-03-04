import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Motion detection parameters
MOTION_THRESHOLD = 2.0  # Value for decent stability

# Start Webcam Feed
cap = cv2.VideoCapture(0)  # 0 = Default Webcam

# Set resolution if input is of poor quality
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
    # pyramid scale(lower for details), levels: large gives large movement, window_size: improves against noise(larger), iterations per pixel (higher is better), 
    # polynomial terms (motion within neighborhood, 5: ast/rough, 7 slow/smooth), 
    # smooth before flow, dense flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Magnitude of flow vector to get how much pixel moves
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Update previous frame
    prev_gray = gray.copy()

    # Get detected objects within the frame
    results = model(frame)
    moving_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Only detected people are relevant
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Motion of object from optical flow matrix
                motion_region = magnitude[y1:y2, x1:x2]
                motion_score = np.mean(motion_region)

                # For object motion beyond threshold
                if motion_score > MOTION_THRESHOLD:
                    moving_objects.append((x1, y1, x2, y2))

    # Bounding box for moving object
    for (x1, y1, x2, y2) in moving_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(frame, "Moving Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display output frame
    cv2.imshow("Realtime YOLO + Optical Flow Motion Filtering", frame)

    # wait / quit
    if cv2.waitKey(10) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
