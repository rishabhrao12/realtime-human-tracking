import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Motion detection threshold
MOTION_THRESHOLD = 2.0  # Controls how much movement is needed to track an object

# Open video file
input_video_path = "input_video.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video frame rate (fps) and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer to save output
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

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

# Function to apply CLAHE for better visibility in low-light conditions
def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)  # Split channels

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # Apply CLAHE to L channel
    l_enhanced = clahe.apply(l)

    enhanced_lab = cv2.merge([l_enhanced, a, b])  # Merge channels back
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)  # Convert back to BGR

    return enhanced_frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply CLAHE to improve visibility in low light
    frame = apply_clahe(frame)

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

    # Write the processed frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
