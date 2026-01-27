import swatahvision as sv

# ---------------------------------------------
# Configure label annotation (text on bounding box)
# ---------------------------------------------
label_annotator = sv.LabelAnnotator(
    color=sv.Color.YELLOW,  # Background color of label
    text_color=sv.Color.BLACK,  # Label text color
    text_position=sv.Position.TOP_LEFT,
    text_scale=0.7,  # Text size
    text_padding=8,  # Padding around text
    smart_position=False,  # Disable auto positioning
)

# ---------------------------------------------
# Configure bounding box annotation
# ---------------------------------------------
box_annotator = sv.BoxAnnotator(sv.Color.YELLOW)

# ---------------------------------------------
# Load SSD Lite MobileNetV3 model
# - OpenVino runtime
# - CPU inference
# ---------------------------------------------
model = sv.Model(
    model="retinanet-resnet50-fpn.xml", engine=sv.Engine.OPENVINO, hardware=sv.Hardware.CPU
)

# ---------------------------------------------
# Load input image from file
# ---------------------------------------------
image = sv.Image.load_from_file("assets/car.jpg")

# ---------------------------------------------
# Run object detection on the image
# ---------------------------------------------
outs = model(image)

# ---------------------------------------------
# Convert model output to detections
# Apply confidence threshold
# ---------------------------------------------
detections = sv.Detections.from_retinanet(outs, conf_threshold=0.5)

# ---------------------------------------------
# Draw labels (class name, confidence, etc.)
# ---------------------------------------------
image = label_annotator.annotate(scene=image, detections=detections)

# ---------------------------------------------
# Draw bounding boxes on the image
# ---------------------------------------------
image = box_annotator.annotate(scene=image, detections=detections)

# ---------------------------------------------
# Display the final annotated image
# ---------------------------------------------
sv.Image.show(image=image)
