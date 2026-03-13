# Age, Gender & Emotion Detection using swatahVision (OpenVINO Engine)

This example demonstrates how to perform **real-time Age, Gender, and Emotion prediction**
using **OpenVINO retail models** through the **swatahVision framework**.

The script captures frames from the **webcam**, detects faces, and predicts:

- **Age**
- **Gender**
- **Emotion**

for each detected face in real time.

The results are displayed directly on the webcam feed.

---

## 📁 Folder Structure

```
emotion_detection_project/
├── emotions-recognition-retail-0003_openvino.py
├── README.md
```

Model files are loaded from the OpenVINO model directory.

---

## 📥 Model Download

Pretrained models used in this example:

| Model | Purpose |
|------|------|
| `age-gender-recognition-retail-0013` | Predicts age and gender |
| `emotions-recognition-retail-0003` | Predicts facial emotion |

These models are part of the **OpenVINO Model Zoo**.

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

Run the script:

```bash
python emotions-recognition-retail-0003_openvino.py
```

Your webcam will open and start detecting:

```
Age
Gender
Emotion
```

Press **ESC** to exit the program.

---

# 📊 Example Output

The webcam window will display bounding boxes with predictions:

```
Male 34 | happy
Female 22 | surprise
Male 45 | neutral
```

Each label contains:

```
Gender Age | Emotion
```

---

# 🧠 Model Information

| Property | Value |
|------|------|
| Framework | swatahVision |
| Engine | OpenVINO |
| Hardware | CPU |
| Age/Gender Model | age-gender-recognition-retail-0013 |
| Emotion Model | emotions-recognition-retail-0003 |
| Input | Face image |
| Output | Age, Gender, Emotion |

---

# ⚠️ Notes

- The webcam must be enabled.
- The model works best with **clear frontal faces**.
- Face detection uses **OpenCV Haar Cascade**.
- Bounding box smoothing is applied to reduce jitter.
- Age prediction is an **estimate**, not exact.

---

# Summary

This project demonstrates a **real-time face analysis system** using swatahVision and OpenVINO.

The pipeline performs:

```
Webcam Frame
      ↓
Face Detection
      ↓
Age & Gender Prediction
      ↓
Emotion Recognition
      ↓
Display Results on Screen
```

This can be used in applications such as:

- Human-computer interaction
- Smart retail analytics
- Emotion-aware systems
- Computer vision demos