# SSD Object Detection using SwatahVision

This example demonstrates how to perform **object detection**
using the **SSD (Single Shot Detector)** model with the
**SwatahVision framework**.

SSD is a fast and efficient deep learning model for detecting
multiple objects in images.

The script loads an image, performs inference using **OpenVINO
through SwatahVision**, and prints detected objects with confidence.

---

## 📥 Model Download

Download the SSD model from the official SwatahVision model repository.

🔗 **Model Repository:**  
[SSD – SwatahVision HuggingFace](https://huggingface.co/swatah/swatahvision/tree/main/detection/ssdlite-mobilenetv3)

Download the following files:

- `ssdlite-mobilenetv3.bin`
- `ssdlite-mobilenetv3.xml`

Place the downloaded model files inside your project directory.

---

## 📁 Project Structure

```
ssd/
│
├── ssdlite-mobilenetv3.bin
├── ssdlite-mobilenetv3.xml
├── ssdlite-mobilenetv3_openvino.py
└── car.jpg
```

---

## ⚙️ Requirements

- Python 3.9+
- NumPy
- OpenCV
- SwatahVision

---

## 📦 Installation

Install the required dependencies:

```bash
pip install numpy
pip install opencv-python
pip install swatahvision
```

---

## 🚀 Running the Example

Run the Python script using:

```bash
ssdlite-mobilenetv3_openvino.py
```

---

## 🧪 Example Code

```python
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
    model="ssdlite-mobilenetv3.xml", engine=sv.Engine.OPENVINO, hardware=sv.Hardware.CPU
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
detections = sv.Detections.from_ssd(outs, conf_threshold=0.5)

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

## 📤 Example Output

```
Car detected (0.89)
```

The model returns:

- Detected object labels
- Bounding box coordinates
- Confidence scores

---

## 👨‍💻 Authors

- **Atharva Kotkar**  
- **Aarav Agrawal**  
- MIT Internship – Swatah AI