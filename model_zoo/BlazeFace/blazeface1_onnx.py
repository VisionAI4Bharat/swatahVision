import swatahVision as sv
import numpy as np

# ---------------------------------------------
# Configure bounding box annotation
# ---------------------------------------------
box_annotator = sv.BoxAnnotator(sv.Color.YELLOW)

# ---------------------------------------------
# Load BlazeFace model
# ---------------------------------------------
model = sv.Model(
    model="blazeface.onnx",
    engine=sv.Engine.ONNX,
    hardware=sv.Hardware.CPU
)

# ---------------------------------------------
# Load image
# ---------------------------------------------
image = sv.Image.load_from_file("assets/sample.jpg")

# ---------------------------------------------
# Run inference
# ---------------------------------------------
outs = model(image)

# ---------------------------------------------
# Extract boxes
# ---------------------------------------------
boxes = outs[0][0]

xyxy = []

# Only parse detections if they exist
if boxes.shape[0] > 0:
    for det in boxes:
        top_y, top_x, bot_y, bot_x = det[:4]
        xyxy.append([top_x, top_y, bot_x, bot_y])

xyxy = np.array(xyxy) if len(xyxy) > 0 else np.empty((0,4))

detections = sv.Detections(
    xyxy=xyxy,
    confidence=np.ones(len(xyxy))
)

# ---------------------------------------------
# Draw bounding boxes
# ---------------------------------------------
image = box_annotator.annotate(scene=image, detections=detections)

# ---------------------------------------------
# Show result
# ---------------------------------------------
sv.Image.show(image=image)