# Pose Estimation using MoveNet (ONNX)
This project demonstrates **human pose estimation** using a MoveNet ONNX model.
It supports image and video inference and provides structured pose outputs (keypoints + confidence) along with visualization.

## 🚀 Features

* MoveNet ONNX inference (CPU)
* Image and video pose estimation
* Keypoints parsing (17 COCO joints)
* Skeleton visualization
* Clean class-based architecture
* Compatible with swatahVision style parsers

---

## 📥 Model Download

Pretrained models for **swatahVision** are available in the Model Zoo.

🔗 [https://visionai4bharat.github.io/swatahVision/model_zoo/](https://visionai4bharat.github.io/swatahVision/model_zoo/)

---

## 📁 Project Structure

```
pose_estimation/
│
├── Movenet.onnx                # Pose estimation model
├── movenet_pose.py             # Inference class (video / image)
├── pose.py                     # Pose parser (from_movenet)
├── assets/
│   └── sample.jpg              # Example input image
└── README.md
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install onnxruntime opencv-python numpy
```

(Optional if using swatahVision)

```bash
pip install swatahVision
```

---

## ⬇️ Download the MoveNet ONNX Model

### Option 1 — Download pre-converted ONNX (recommended)

1. Download MoveNet ONNX model from a model hub (example sources):

   * ONNX model zoo mirrors
   * TensorFlow → ONNX community conversions
   * Your internal model storage

2. Rename the file to:

```
Movenet.onnx
```

3. Place it inside:

```
pose_estimation/
```

---

### Option 2 — Convert MoveNet to ONNX yourself

If you have TensorFlow MoveNet:

Install:

```bash
pip install tf2onnx tensorflow
```

Convert:

```bash
python -m tf2onnx.convert \
--saved-model movenet_saved_model \
--output Movenet.onnx \
--opset 13
```

Move the generated file into the project folder.

---

## ▶️ Run Pose Estimation (Image)

Example:

```python
import swatahVision as sv
from pose import Pose

model = sv.Model(
    model="pose_estimation/Movenet.onnx",
    engine=sv.Engine.ONNX,
    hardware=sv.Hardware.CPU
)

image = sv.Image.load_from_file("pose_estimation/assets/sample.jpg")

outs = model(image)

pose = Pose.from_movenet(outs)

print(pose.keypoints)
print(pose.confidence)
```

---

## ▶️ Run Pose Estimation (Video)

```bash
python movenet_pose.py
```

Press **q** to exit.

---

## 🧠 Output Format

MoveNet returns 17 keypoints:

```
[x, y] coordinates
confidence score
```

Keypoints follow COCO order:

* nose
* eyes
* ears
* shoulders
* elbows
* wrists
* hips
* knees
* ankles

---

## 🛠️ How it works

Pipeline:

1. Load ONNX model
2. Preprocess frame (resize → normalize)
3. Run ONNX inference
4. Parse keypoints via `Pose.from_movenet`
5. Draw skeleton

---

## ✅ Supported Tasks

* Pose estimation
* Real-time video pose
* Visualization
* Framework integration (swatahVision)

---

## 📌 Notes

* Model expects **192×192 input**
* Works on CPU
* Confidence threshold can be adjusted
* Compatible with other pose ONNX models with similar output

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch
3. Make changes
4. Open Pull Request

----