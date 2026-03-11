# RetinaNet Object Detection using swatahVision

This example demonstrates how to perform **object detection**
using the **RetinaNet model** through the **swatahVision framework**.

The model detects objects in an image and returns **bounding boxes,
class labels, and confidence scores**.

---

## 📁 Folder Structure

```
retinanet_example/
├── retinanet.xml
├── retinanet.bin
├── retinanet-resnet50-fpn_openvino.py
└── assets/
    └── car.jpg
```

---

## 📥 Model Download

Pretrained models for **swatahVision** are available in the Model Zoo.

🔗 [https://visionai4bharat.github.io/swatahVision/model_zoo/](https://visionai4bharat.github.io/swatahVision/model_zoo/)


Required files:

- `retinanet.xml`
- `retinanet.bin`

---

## 🔧 Requirements

- Python 3.9+
- NumPy
- OpenCV
- swatahVision

---

## 🧩 Installation

```bash
pip install numpy
pip install opencv-python
pip install swatahVision
```

---

## 🚀 How to Run

```bash
retinanet-resnet50-fpn_openvino.py
```

---

# 🧪 Example Code

```python
import swatahVision as sv

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
```

---

# 📤 Output

The model returns:

- Bounding box coordinates
- Class labels
- Confidence score

Example:

```

Detected: Car (0.89)
```

---