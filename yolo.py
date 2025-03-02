from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # Using Nano version for real-time speed

# Load the video
video_path = "trial2.mp4"
cap = cv2.VideoCapture(video_path)

# Get video details
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Define Video Writer to save output
output_path = "output_persons2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frames left

    # Run YOLO detection
    results = model(frame)

    for result in results:
        boxes = result.boxes  # Get bounding boxes

        for box in boxes:
            if int(box.cls) == 0:  # Filter only "person" class (COCO ID = 0)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to output video
    out.write(frame)

    # Show the frame (optional, can remove if running on a server)
    cv2.imshow("YOLO Person Detection", frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_path}")
