<<<<<<< HEAD
<<<<<<< HEAD
# YOLO Object Detection using swatahvision

This example demonstrates how to perform **object detection**
using the **YOLO model** with the **swatahvision framework**.
=======
# YOLO Object Detection using swatahVision

This example demonstrates how to perform **object detection**
using the **YOLO model** with the **swatahVision framework**.
>>>>>>> a64a1fb (doc 26.03.01rc1)
=======
# YOLO Object Detection using swatahVision

This example demonstrates how to perform **object detection**
using the **YOLO model** with the **swatahVision framework**.
>>>>>>> bd163acb0252bcd1e9cb9d5690171f43e3ee9260

YOLO (**You Only Look Once**) is a real-time object detection model
that detects multiple objects in a single forward pass of the network.

<<<<<<< HEAD
<<<<<<< HEAD
The script loads an image, runs inference using **OpenVINO through swatahvision**,
=======
The script loads an image, runs inference using **OpenVINO through swatahVision**,
>>>>>>> a64a1fb (doc 26.03.01rc1)
=======
The script loads an image, runs inference using **OpenVINO through swatahVision**,
>>>>>>> bd163acb0252bcd1e9cb9d5690171f43e3ee9260
and prints the detected objects with their confidence scores.

---

## 📥 Model Download

<<<<<<< HEAD
<<<<<<< HEAD
Download the YOLO model from the official swatahvision model repository.

🔗 **Model Link:**  
[YOLOv8 – swatahvision HuggingFace](https://huggingface.co/swatah/swatahvision/tree/main/detection/yolov8)
=======
Pretrained models for **swatahVision** are available in the Model Zoo.

🔗 [https://visionai4bharat.github.io/swatahVision/model_zoo/](https://visionai4bharat.github.io/swatahVision/model_zoo/)
>>>>>>> a64a1fb (doc 26.03.01rc1)
=======
Pretrained models for **swatahVision** are available in the Model Zoo.

🔗 [https://visionai4bharat.github.io/swatahVision/model_zoo/](https://visionai4bharat.github.io/swatahVision/model_zoo/)
>>>>>>> bd163acb0252bcd1e9cb9d5690171f43e3ee9260

Download the following files:

- `yolov8n.bin`
- `yolov8n.xml`

Place the downloaded model files inside your project directory.

---

## 📁 Project Structure

```
yolo_example/
│
├── yolov8n.bin
├── yolov8n.xml
├── yolov8n_openvino.py
└── car.jpg
```

---

## ⚙️ Requirements

- Python 3.9+
- NumPy
- OpenCV
<<<<<<< HEAD
<<<<<<< HEAD
- swatahvision
=======
- swatahVision
>>>>>>> a64a1fb (doc 26.03.01rc1)
=======
- swatahVision
>>>>>>> bd163acb0252bcd1e9cb9d5690171f43e3ee9260

---

## 📦 Installation

Install the required dependencies:

```bash
pip install numpy
pip install opencv-python
pip install swatahVision
```

---

## 🚀 Running the Example

Run the Python script using:

```bash
yolov8n_openvino.py
```

---

## 🧪 Example Code

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
# Load Yolo8n model
# - ONNX runtime
# - CPU inference
# ---------------------------------------------
model = sv.Model(
    model="yolov8n.xml", engine=sv.Engine.OPENVINO, hardware=sv.Hardware.CPU
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
detections = sv.Detections.from_yolo(outs, conf_threshold=0.5)

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
Car detected (0.88)
```

The model returns:

- Detected object labels
- Bounding box coordinates
- Confidence scores

---