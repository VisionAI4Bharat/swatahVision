import cv2
import swatahvision as sv

# ---------------------------------------------
# Initialize BYTETracker for multi-object tracking
# frame_rate should match the video FPS
# ---------------------------------------------
tracker = sv.ByteTrack(frame_rate=30)

# ---------------------------------------------
# Load RetinaNet object detection model
# - Using ONNX engine
# - Running on GPU
# ---------------------------------------------
model = sv.Model(
    model="retinanet-resnet50-fpn-13",
    engine=sv.Engine.ONNX,
    hardware=sv.Hardware.GPU,
)

# ---------------------------------------------
# Bounding box annotator configuration
# ---------------------------------------------
box_annotator = sv.BoxAnnotator(sv.Color.YELLOW)

# ---------------------------------------------
# Label annotator configuration
# ---------------------------------------------
label_annotator = sv.LabelAnnotator(
    color=sv.Color.YELLOW,           # Label background color
    text_color=sv.Color.BLACK,        # Text color
    text_position=sv.Position.TOP_LEFT,
    text_scale=0.5,
    text_padding=8,
    smart_position=False,
)

# ---------------------------------------------
# Load input video
# ---------------------------------------------
video_path = "tests/assets/people-walking.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# ---------------------------------------------
# Video processing loop
# ---------------------------------------------
while True:
    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        break

    # -----------------------------------------
    # Run object detection on current frame
    # -----------------------------------------
    results = model(frame)

    # -----------------------------------------
    # Convert model output to Detections object
    # Apply confidence threshold
    # -----------------------------------------
    detections = sv.Detections.from_ssd(
        results,
        conf_threshold=0.3
    )

    # -----------------------------------------
    # Update tracker with current detections
    # Assigns unique tracker IDs
    # -----------------------------------------
    tracked_objects = tracker.update_with_detections(detections)

    # -----------------------------------------
    # Create labels using tracker IDs
    # -----------------------------------------
    labels = [
        f"ID {tracker_id}" for tracker_id in detections.tracker_id
    ]

    # -----------------------------------------
    # Draw bounding boxes
    # -----------------------------------------
    frame = box_annotator.annotate(
        scene=frame,
        detections=detections
    )

    # -----------------------------------------
    # Draw labels on top of bounding boxes
    # -----------------------------------------
    frame = label_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

    # -----------------------------------------
    # Display output frame
    # -----------------------------------------
    cv2.imshow("BYTETracker", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------------
# Release resources
# ---------------------------------------------
cap.release()
cv2.destroyAllWindows()
