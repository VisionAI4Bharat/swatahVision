import swatahVision as sv
import numpy as np
import cv2

MODEL_PATH = "emotions-recognition-retail-0003.xml"
IMAGE_PATH_1 = "image_happy.jpg"
IMAGE_PATH_2 = "image_sad.jpg"

model = sv.Model(
    model=MODEL_PATH,
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)

# Emotion labels
emotions = ["neutral", "happy", "sad", "surprise", "anger"]

# -------------------------
# First Image
# -------------------------
img = sv.Image.load_from_file(IMAGE_PATH_1)

outputs = model(img)[0]

emotion_id = int(np.argmax(outputs))
emotion = emotions[emotion_id]

print("Predicted Emotion:", emotion)


# -------------------------
# Second Image
# -------------------------
image = sv.Image.load_from_file(IMAGE_PATH_2)

outputs = model(image)[0]

emotion_id = int(np.argmax(outputs))
emotion = emotions[emotion_id]

print("Predicted Emotion:", emotion)