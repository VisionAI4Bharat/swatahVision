import swatahvision as sv
import numpy as np

# ---------------------------------------------
# Configure label annotation (text on bounding box)
# ---------------------------------------------
label_annotator = sv.LabelAnnotator(
    color=sv.Color.YELLOW,
    text_color=sv.Color.BLACK,
    text_position=sv.Position.TOP_LEFT,
    text_scale=0.7,
    text_padding=8,
    smart_position=False,
)

# ---------------------------------------------
# Configure bounding box annotation
# ---------------------------------------------
box_annotator = sv.BoxAnnotator(sv.Color.YELLOW)

# ---------------------------------------------
# Load face detection ONNX model
# ---------------------------------------------
model = sv.Model(
    model="face_detection.onnx",  # path to ONNX model
    engine=sv.Engine.ONNX,
    hardware=sv.Hardware.CPU
)

# ---------------------------------------------
# Load input image
# ---------------------------------------------
image = sv.Image.load_from_file("assets/face.jpg")

# ---------------------------------------------
# Run inference
# ---------------------------------------------
outs = model(image)

# ---------------------------------------------
# Convert model output to detections
# ---------------------------------------------
detections = sv.Detections.from_ssd(
    outs,
    conf_threshold=0.5
)

# ---------------------------------------------
# Draw labels
# ---------------------------------------------
image = label_annotator.annotate(
    scene=image,
    detections=detections
)

# ---------------------------------------------
# Draw bounding boxes
# ---------------------------------------------
image = box_annotator.annotate(
    scene=image,
    detections=detections
)

# ---------------------------------------------
# Convert image to numpy for cropping
# ---------------------------------------------
img_np = np.array(image)

# ---------------------------------------------
# Crop detected faces
# ---------------------------------------------
faces = []

for box in detections.xyxy:
    x1, y1, x2, y2 = map(int, box)

    face_crop = img_np[y1:y2, x1:x2]
    faces.append(face_crop)

# ---------------------------------------------
# Show annotated image
# ---------------------------------------------
sv.Image.show(image=image)

# ---------------------------------------------
# Show cropped faces
# ---------------------------------------------
for i, face in enumerate(faces):
    sv.Image.show(image=sv.Image(face))