# Age & Gender Prediction using OpenVINO

This example demonstrates how to perform **Age and Gender prediction**
using the **age-gender-recognition-retail-0013** model with  
**OpenVINO Runtime** through the **SwatahVision** framework.

The script loads a **cropped face image**, runs inference on CPU,
and prints the predicted **age** and **gender** in the terminal.

---

## 📁 Folder Structure


examples/
└── age_gender_openvino/
├── age-gender-recognition-retail-0013_openvino.py
├── README.md
└── assets/
└── face.jpg

---

## 🖼 Required Input Image

- Add **one cropped face image** inside the `assets/` folder
- Example file name:
  assets/face.jpg

- The image **must contain only one face**
- Best results are obtained when the face is:
- Front-facing
- Well-lit
- Clearly visible

You may use **any face image** (male or female).

---

## 🔧 Requirements

- Python **3.10**
- OpenVINO **2024.6**
- NumPy
- OpenCV
- SwatahVision

#Install dependencies:
pip install openvino numpy opencv-python
pip install git+https://github.com/VisionAI4Bharat/swatahVision.git

##🚀 How to Run
Navigate to the example folder and run:
python age-gender-recognition-retail-0013_openvino.py

##📤 Output
The script prints the predicted age and gender:
🎯 Predicted Age   : 24
🎯 Predicted Gender: Male

Age is an estimated value
Gender is predicted as Male / Female

🧠 Model Information

Model Name: age-gender-recognition-retail-0013
Framework: OpenVINO
Input Size: 62 × 62
Outputs:
Age → normalized value (multiplied by 100)
Gender → probabilities [Female, Male]

⚠️ Notes

This example expects a cropped face, not a full image
The model supports single-face inference
Predictions may vary based on face quality

👨‍💻 Author
-Atharva Kotkar
-Aarav Agarwal 
-MIT Internship – Swatah AI
