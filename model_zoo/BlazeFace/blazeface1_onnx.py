import swatahVision as sv
import numpy as np

# ---------------------------------------------
# Configure bounding box annotation
# ---------------------------------------------
box_annotator = sv.BoxAnnotator(sv.Color.YELLOW)

# ---------------------------------------------
# Load BlazeFace model
# - ONNX runtime
# - CPU inference
# ---------------------------------------------
model = sv.Model(
    model="C:\\Users\\LENOVO\\Downloads\\blaze_fixed.onnx",
    engine=sv.Engine.ONNX,
    hardware=sv.Hardware.CPU
)

# ---------------------------------------------
# Load input image
# ---------------------------------------------
image = sv.Image.load_from_file("C:\\Users\\LENOVO\\Downloads\\example1.png")

# ---------------------------------------------
# Run face detection
# ---------------------------------------------
outs = model(image)

# ---------------------------------------------
# Parse BlazeFace outputs
# ---------------------------------------------
boxes = outs[0][0]

if boxes.shape[1] == 0:
    print("No faces detected")
    detections = sv.Detections.empty()

else:
    boxes = boxes[0]

    xyxy = []

    for det in boxes:
        top_y, top_x, bot_y, bot_x = det[:4]
        xyxy.append([top_x, top_y, bot_x, bot_y])

    xyxy = np.array(xyxy)

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=np.ones(len(xyxy))
    )

# ---------------------------------------------
# Draw bounding boxes
# ---------------------------------------------
image = box_annotator.annotate(scene=image, detections=detections)

# ---------------------------------------------
# Display result
# ---------------------------------------------
sv.Image.show(image=image)