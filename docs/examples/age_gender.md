# Age & Gender Prediction using SwatahVision (OpenVINO Engine)

This example demonstrates how to perform **Age and Gender prediction**
using the **age-gender-recognition-retail-0013** model through the  
**SwatahVision framework**, powered internally by the **OpenVINO engine**.

The script loads two face images, runs inference on **CPU**,  
and prints the predicted **age** and **gender** in the terminal.

---

## 📁 Folder Structure

```
age_gender_project/
├── age-gender-recognition-retail-0013.xml
├── age-gender-recognition-retail-0013.bin
├── age_gender_prediction.py
├── README.md
├── image_old_man.jpg
└── image_boy.jpg
```

---

## 📥 Model Download

Download the model from Hugging Face:

🔗 **Model Repository:**  
[https://huggingface.co/swatah/swatahvision/tree/main/classifiation/age-gender-recognition-retail-0013](https://huggingface.co/swatah/swatahvision/tree/main/classifiation/age-gender-recognition-retail-0013)

Download the following files:

- `age-gender-recognition-retail-0013.xml`
- `age-gender-recognition-retail-0013.bin`

Place both files inside your project root directory.

---

## 🔧 Requirements

- Python 3.9+
- NumPy
- OpenCV
- SwatahVision

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
pip install swatahvision
```

---

## 🚀 How to Run

```bash
python age_gender_prediction.py
```

---

# 🧪 Complete Source Code

Below is the exact code used in this project:

```python
import swatahvision as sv
import numpy as np
import cv2

MODEL_PATH = "age-gender-recognition-retail-0013.xml"
IMAGE_PATH_OLD = "image_old_man.jpg"
IMAGE_PATH_BOY = "image_boy.jpg"

model = sv.Model(
    model=MODEL_PATH,
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)

# -----------------------------
# First Image Prediction
# -----------------------------
img = sv.Image.load_from_file(IMAGE_PATH_OLD)
outputs = model(img)[0]

gender_blob = outputs[0]
age_blob = outputs[1]

gender_id = int(np.argmax(gender_blob))
gender = "Male" if gender_id == 1 else "Female"
age = int(age_blob[0][0][0][0] * 100)

print("Predicted Age   :", age)
print("Predicted Gender:", gender)

# -----------------------------
# Second Image Prediction
# -----------------------------
image = sv.Image.load_from_file(IMAGE_PATH_BOY)
outputs = model(image)[0]
print(outputs)

gender_blob = outputs[0]
age_blob = outputs[1]

gender_id = int(np.argmax(gender_blob))
gender = "Male" if gender_id == 1 else "Female"
age = int(age_blob[0][0][0][0] * 100)

print("Predicted Age   :", age)
print("Predicted Gender:", gender)
```

---

# 📤 Sample Output

```
Predicted Age   : 62
Predicted Gender: Male

Predicted Age   : 14
Predicted Gender: Male
```

---

# 🧠 Model Information

- **Model Name:** `age-gender-recognition-retail-0013`
- **Framework:** SwatahVision
- **Inference Engine:** OpenVINO (internal)
- **Hardware:** CPU
- **Age Output:** Normalized value × 100
- **Gender Output:** Probability scores `[Female, Male]`

---

# ⚠️ Notes

- The model expects cropped face images.
- Only single-face inference is supported.
- Age prediction is an estimate.
- Accuracy depends on lighting and image quality.

---

# 👨‍💻 Authors

- **Atharva Kotkar**  
- **Aarav Agarwal**  
- MIT Internship – Swatah AI