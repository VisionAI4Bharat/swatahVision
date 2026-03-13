import swatahVision as sv
import numpy as np
import cv2

AGE_MODEL = "intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml"
EMOTION_MODEL = "intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml"

age_gender_model = sv.Model(
    model=AGE_MODEL,
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)

emotion_model = sv.Model(
    model=EMOTION_MODEL,
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)

emotions = ["neutral","happy","sad","surprise","anger"]

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

frame_count = 0
faces = []

prev_x, prev_y, prev_w, prev_h = 0,0,0,0
alpha = 0.7

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_count += 1

    # Detect every 5 frames
    if frame_count % 5 == 0:

        detected = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(120,120)
        )

        # If detection succeeds, update faces
        if len(detected) > 0:
            faces = detected

    for (x,y,w,h) in faces:

        # Smooth bounding box
        x = int(alpha * prev_x + (1-alpha) * x)
        y = int(alpha * prev_y + (1-alpha) * y)
        w = int(alpha * prev_w + (1-alpha) * w)
        h = int(alpha * prev_h + (1-alpha) * h)

        prev_x, prev_y, prev_w, prev_h = x,y,w,h

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        cv2.imwrite("face.jpg", face)

        image = sv.Image.load_from_file("face.jpg")

        # Age + Gender
        outputs = age_gender_model(image)[0]

        gender_blob = outputs[0]
        age_blob = outputs[1]

        gender_id = int(np.argmax(gender_blob))
        gender = "Male" if gender_id == 1 else "Female"

        age = int(age_blob[0][0][0][0] * 100)

        # Emotion
        emo_out = emotion_model(image)[0]

        emotion_id = int(np.argmax(emo_out))
        emotion = emotions[emotion_id]

        label = f"{gender} {age} | {emotion}"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(
            frame,
            label,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Age Gender Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()