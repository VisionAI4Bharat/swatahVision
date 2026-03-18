# BlazeFace Face Detection using swatahVision (ONNX Engine)

This example demonstrates how to perform **real-time face detection**
using **BlazeFace model** with the **swatahVision framework**.

The script supports both::

- **Webcam (real-time detection)**
- **Video file processing**

For each detected face, it provides:

- Bounding boxes
- Confidence score
- Facial landmarks (eyes, nose, mouth, ears)

The results are displayed directly on the screen.

---

## 📁 Folder Structure

```
face_detection/
    ├── blazeface_onnx.py
    ├── README.md
```

---

## 📥 Model Download

This example uses:

| Model | Purpose |
|------|------|
| `blazeface.onnx` | Fast face detection with landmarks |

BlazeFace is a lightweight and fast face detector designed for real-time applications.

More models are available here:

🔗 https://visionai4bharat.github.io/swatahVision/model_zoo/

---

## 🔧 Requirements

- Python 3.9+
- NumPy
- OpenCV
- swatahVision

---

## 🧩 Installation

### Create Environment (Recommended)

```bash
conda create -n swatah_env python=3.9 -y
conda activate swatah_env
```

### Install Dependencies

```bash
pip install numpy
pip install opencv-python
pip install swatahVision
```

---

## 🚀 How to Run

▶️ Run Webcam

```bash
python blazeface_onnx.py
```
▶️ Run Video

Edit the script:
```bash
model.run_video("your_video_path.mp4")
```
Then run:
```bash
python blazeface_onnx.py
```

Press **ESC** to exit.

# 📊 Example Output

The output window will display:
- Face bounding boxes
- Confidence scores
- Facial landmarks

Example:
- 0.92
- 0.87
- 0.76

Each face shows:
- Bounding Box + Confidence + Landmarks

---

# 🧠 Model Information

| Property | Value |
|------|------|
| Framework | swatahVision |
| Engine | ONNX |
| Hardware | CPU |
| Model | BlazeFace |
| Input Size | 128 x 128 |
| Output | Bounding Box + 6 Landmarks |
| Inference Type | Real-time |

---

# ⚙️ Pipeline

```
Frame (Webcam/Video)
      ↓
Preprocessing (Resize + Normalize)
      ↓
BlazeFace ONNX Inference
      ↓
Postprocessing (Boxes + Scores)
      ↓
Landmark Extraction
      ↓
Draw Results (Box + Score + Landmarks)
      ↓
Display Output

```
---

# ⚠️ Notes

- The model requires 128×128 input preprocessing
- Manual preprocessing is used instead of default swatahVision preprocessing
- Landmark points are drawn manually (not part of sv.Detections)
- Works best with clear frontal faces
- Confidence threshold can be adjusted in code


---

# Summary

This project demonstrates a **real-time face detection** system using BlazeFace and swatahVision.

Features include:

- Fast face detection ⚡
- Confidence scoring 📊
- Facial landmark detection 🎯
- Webcam + video support 🎥

This can be used in applications such as:

- Face tracking systems
- Real-time vision applications
- Mobile/edge AI demos
- Computer vision learning projects