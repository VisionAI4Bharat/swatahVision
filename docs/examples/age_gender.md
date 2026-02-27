# Age & Gender Prediction using SwatahVision (OpenVINO Backend)

This example demonstrates how to perform *Age and Gender prediction*
using the *age-gender-recognition-retail-0013* model through the  
*SwatahVision framework, with **OpenVINO as the inference engine*.

The script loads *face images, runs inference on **CPU*,  
and prints the predicted *age* and *gender* in the terminal.



## 🖼 Required Input Images

- Add *cropped face images* inside the assets/ folder
- Example file names:
  - assets/image_old_man.jpg
  - assets/image_boy.jpg

### Image Guidelines

- Each image *must contain only one face*
- Best results when the face is:
  - Front-facing
  - Well-lit
  - Clearly visible
  - Properly cropped

You may use *any face images* (male or female).

---

## 🔧 Requirements

- Python *3.9+*
- NumPy
- OpenCV
- SwatahVision

> ⚠️ OpenVINO is used *internally by SwatahVision*  
> You do *NOT* need to write OpenVINO code directly.

---

## 🧩 Installation

### 1️⃣ Create and Activate Conda Environment (Recommended)


conda create -n swatah_env python=3.9 -y
conda activate swatah_env
2️⃣ Install Required Dependencies
pip install numpy
pip install opencv-python
pip install swatahvision
🚀 How to Run

Navigate to the project directory and run:

python age_gender_prediction.py
🧪 Code Explanation
Import Required Libraries
import swatahvision as sv

```bash
import numpy as np
import cv2
Load Age & Gender Model
model = sv.Model(
    model="age-gender-recognition-retail-0013.xml",
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)

sv.Engine.OPENVINO → Uses OpenVINO internally

sv.Hardware.CPU → Runs inference on CPU

Run Inference on Image
image = sv.Image.load_from_file("assets/image_old_man.jpg")
outputs = model(image)[0]
Extract Model Outputs
gender_blob = outputs[0]
age_blob = outputs[1]

gender_blob → Gender probabilities

age_blob → Normalized age value

Process Gender Output
gender_id = int(np.argmax(gender_blob))
gender = "Male" if gender_id == 1 else "Female"
Process Age Output
age = int(age_blob[0][0][0][0] * 100)

The model outputs age as a normalized value (0–1)
It is multiplied by 100 to get age in years.

##📤 Output

The script prints the predicted age and gender in the terminal:

Predicted Age   : 62
Predicted Gender: Male

Predicted Age   : 14
Predicted Gender: Male
##🧠 Model Information

Model Name: age-gender-recognition-retail-0013

Inference Framework: SwatahVision

Inference Engine: OpenVINO (internal)

Input Size: 62 × 62

Outputs:

Age → normalized regression value × 100

Gender → probabilities [Female, Male]

##⚠️ Notes

This example expects cropped face images

Only single-face inference is supported

Prediction accuracy depends on:

Face alignment

Lighting conditions

Image quality

Age prediction is approximate, not exact

###👨‍💻 Author

Aarav Agarwal

Atharva Kotkar

MIT Internship – Swatah AI