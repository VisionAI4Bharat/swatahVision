import cv2
import numpy as np
from openvino.runtime import Core

# ----------------------------
# CONFIG
# ----------------------------
MODEL_XML = "age-gender-recognition-retail-0013.xml"
IMAGE_PATH = "assets/image.png"

# ----------------------------
# Load OpenVINO model
# ----------------------------
core = Core()
model = core.read_model(MODEL_XML)
compiled_model = core.compile_model(model, "CPU")

input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

# ----------------------------
# Load & preprocess image
# ----------------------------
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("Image not found. Check IMAGE_PATH")

img = cv2.resize(img, (62, 62))
img = img.transpose(2, 0, 1)      # HWC → CHW
img = img.reshape(1, 3, 62, 62)

# ----------------------------
# Inference
# ----------------------------
results = compiled_model([img])

gender_blob = results[output_layers[0]]  # shape (1,2,1,1)
age_blob = results[output_layers[1]]     # shape (1,1,1,1)

# ----------------------------
# Post-processing
# ----------------------------
gender_id = int(np.argmax(gender_blob))
gender = "Male" if gender_id == 1 else "Female"

age = int(age_blob[0][0][0][0] * 100)

# ----------------------------
# Output
# ----------------------------
print("🎯 Predicted Age   :", age)
print("🎯 Predicted Gender:", gender)