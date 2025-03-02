from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort  # Import SORT tracker

# Load the pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # YOLOv8 Nano for real-time speed

# Initialize SORT tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Load the video
video_path = "trial2.mp4"
cap = cv2.VideoCapture(video_path)

# Get video details
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Define Video Writer to save output
output_path = "output_tracked.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frames left

    # Run YOLO detection
    results = model(frame)

    detections = []  # List to store YOLO detections for SORT

    for result in results:
        boxes = result.boxes  # Get bounding boxes

        for box in boxes:
            if int(box.cls) == 0:  # Filter only "person" class (COCO ID = 0)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = float(box.conf[0])  # Confidence score

                # Append detection in SORT format [x1, y1, x2, y2, confidence]
                detections.append([x1, y1, x2, y2, confidence])

    # Convert detections to NumPy array
    detections = np.array(detections)

    # Run SORT tracker
    if len(detections) > 0:
        tracked_objects = tracker.update(detections)
    else:
        tracked_objects = []

    # Draw bounding boxes and IDs on frame
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)  # Extract SORT tracking results

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write the processed frame to output video
    out.write(frame)

    # Show the frame (optional, can remove if running on a server)
    cv2.imshow("YOLO + SORT Tracking", frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_path}")
