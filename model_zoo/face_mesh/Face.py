import cv2
from ultralytics import YOLO

# Load YOLO pose model
model = YOLO("yolov8n-pose.pt")   # lightweight model (~20MB)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Face Mesh / Landmarks", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()