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

img = sv.Image.load_from_file(IMAGE_PATH_OLD)
outputs = model(img)[0]

gender_blob = outputs[0]
age_blob = outputs[1]

gender_id = int(np.argmax(gender_blob))
gender = "Male" if gender_id == 1 else "Female"
age = int(age_blob[0][0][0][0]* 100)

print("Predicted Age   :", age)
print("Predicted Gender:", gender)


image = sv.Image.load_from_file(IMAGE_PATH_BOY)
outputs = model(image)[0]
print(outputs)

gender_blob = outputs[0]
age_blob = outputs[1]

gender_id = int(np.argmax(gender_blob))
gender = "Male" if gender_id == 1 else "Female"
age = int(age_blob[0][0][0][0]* 100)

print("Predicted Age   :", age)
print("Predicted Gender:", gender)

