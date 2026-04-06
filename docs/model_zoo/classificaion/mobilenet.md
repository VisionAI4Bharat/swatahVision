# Image Classification using MobileNetV2 (OpenVINO Engine)

This example demonstrates how to perform **image classification**
using the **MobileNetV2** model through the  
**swatahVision framework**, powered internally by the **OpenVINO engine**.

The script loads an input image, runs inference on **CPU**,  
and prints the **top-5 predicted classes** in the terminal.

---

## 📁 Folder Structure

```
mobilenet_classification/
├── mobilenetv2.xml
├── mobilenetv2.bin
├── mobilenet_classification.py
├── README.md
└── assets/
    └── car.jpg
```

---

## 📥 Model Download

Pretrained models for **swatahVision** are available in the Model Zoo.

🔗 [https://visionai4bharat.github.io/swatahVision/model_zoo/](https://visionai4bharat.github.io/swatahVision/model_zoo/)

Download the following files:

- `mobilenetv2.xml`
- `mobilenetv2.bin`

Place both files inside your project root directory:

```
mobilenet_classification/
├── mobilenetv2.xml
├── mobilenetv2.bin
```

---

## 🖼 Required Input Image

- Add an input image inside the `assets/` folder  
- Example: `assets/car.jpg`

You may use **any object image** for classification.

---

## 🔧 Requirements

- Python 3.9+
- NumPy
- OpenCV
- swatahVision  

> OpenVINO is used internally by swatahVision.  
> You do **NOT** need to write OpenVINO code manually.

---

## 🧩 Installation

### 1️⃣ Create Environment (Recommended)

```bash
conda create -n swatah_env python=3.9 -y
conda activate swatah_env
```

### 2️⃣ Install Dependencies

```bash
pip install numpy
pip install opencv-python
pip install swatahVision
```

---

## 🚀 How to Run

```bash
python mobilenet_classification.py
```

---

# 🧪 Complete Source Code

```python
import swatahVision as sv

model = sv.Model(
    model="mobilenetv2.xml",
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)

image = sv.Image.load_from_file("assets/car.jpg")

outs = model(image)

classification = sv.Classification.from_mobilenet(outs, top_k=5)

print(classification)
```

---

# 🧠 Model Information

- **Model Name:** MobileNetV2  
- **Framework:** swatahVision  
- **Inference Engine:** OpenVINO (internal)  
- **Hardware:** CPU  
- **Task:** Image Classification  
- **Output:** Class probabilities  
- **Top-K Predictions:** 5  

---

# ⚠️ Notes

- Input images can contain any object.  
- Image quality affects prediction accuracy.  
- Model is trained on ImageNet classes.  
- Output probabilities are normalized scores.