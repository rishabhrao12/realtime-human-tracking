import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Import ByteTrack tracker
from yolox.tracker.byte_tracker import BYTETracker

class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.5  # Confidence threshold for tracking
        self.track_buffer = 100  # Number of frames to keep lost tracks
        self.match_thresh = 0.8  # IoU threshold for association
        self.mot20 = False  # Set to True if using the MOT20 dataset

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy
print(np.__version__)
#"""
# Initialize ByteTrack

args = TrackerArgs()
tracker = BYTETracker(args)

# Load the video
video_path = "Video/trial2.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Define Video Writer
output_path = "output_bytrack.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frames left

    # Run YOLO detection
    results = model(frame)

    detections = []  # List to store YOLO detections for ByteTrack

    for result in results:
        boxes = result.boxes  # Get bounding boxes

        for box in boxes:
            if int(box.cls) == 0:  # Filter only "person" class (COCO ID = 0)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = float(box.conf[0])  # Confidence score

                # Append detection in ByteTrack format [x1, y1, x2, y2, confidence]
                detections.append([x1, y1, x2, y2, confidence])

    # Convert detections to NumPy array
    detections = np.array(detections, dtype=np.float32)

    # Run ByteTrack tracker
    if len(detections) > 0:
        tracked_objects = tracker.update(detections, frame.shape[:2], frame.shape[:2])
    else:
        tracked_objects = []

    # Draw bounding boxes and IDs on frame
    # Draw bounding boxes and IDs on frame
    for track in tracked_objects:
        x1, y1, x2, y2 = map(int, track.tlbr)  # Extract bbox coordinates
        track_id = int(track.track_id)  # Extract track ID

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    # Write the processed frame to output video
    out.write(frame)

    # Show the frame (optional)
    cv2.imshow("YOLO + ByteTrack Tracking", frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_path}")