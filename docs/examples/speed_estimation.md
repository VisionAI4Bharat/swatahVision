# Speed Estimation using YOLO (ONNX) + ByteTrack

This example demonstrates how to estimate the speed of moving objects in a video using computer vision.

The system detects objects in each frame, tracks them across frames, and calculates their speed based on the distance they travel over time.

Speed estimation is widely used in intelligent video analytics systems to analyze motion patterns and monitor moving objects.

This project is ideal for:

- 🚗 Traffic speed monitorin
- 🏙 Smart city surveillance systems
- 🚦 Traffic behavior analysis
- 🏭 Industrial vehicle monitoring
- 📊 Motion analytics in security footage

---

## 🚀 What This Project Does

The system performs the following steps:

1. Detects objects using a YOLO ONNX model
2. Tracks objects across frames using ByteTrack
3. Measures displacement of tracked objects
4. Calculates speed using frame timing
5. Displays estimated speed on the video
6. Saves the annotated output video

Each zone displays the number of people currently inside it.

Example label:

```
#ID SPEED
```
Example:

```
#4 38 km/h
```

This means object ID 4 is moving at 38 km/h.

---

## 📁 Project Structure

```
speed_estimation/
├── yolov11x-1280_onnx.py        # Main script for detection, tracking, and speed estimation
├── models/
|   └── yolov11x-1280.onnx       # YOLO ONNX detection model
├── data/
│   └── input.mp4                # Example input video
└── README.md                    # Example documentation
```

---

## 📁 Project Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/VisionAI4Bharat/swatahvision.git
cd swatahvision/examples/speed_estimation
```

---

### 2️⃣ (Optional) Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```


---

## 🧠 Run Speed Estimation

Run the detection and tracking script:

```bash
python yolov11x-1280_onnx.py \
    --source_video_path "data/input.mp4" \
    --source_weights_path "models/yolov11x-1280.onnx" \
    --classes 2 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

---

## ⚙️ Command Line Arguments

### Required

`--source_video_path`  
Path to input video file.

---

### Optional

`--source_weights_path`  
Path to YOLO ONNX model file.

`--classes`  
List of class IDs to detect.  
Example:

```bash
--classes 2
```
Common class IDs:
- 0 → Person
- 2 → Car
- 5 → Bus

If empty, all detected classes are tracked.

`--confidence_threshold`
Detection confidence threshold.  
Default: `0.3`

`--iou_threshold`  
IoU threshold for Non-Max Suppression.  
Default: `0.7`

---

## 🧠 How It Works (Technical Overview)

Processing Pipeline:

1. Load YOLO ONNX detection model
2. Read frames from the input video
3. Run object detection on each frame
4. Filter detections for selected classes
5. Track objects using ByteTrack multi-object tracking
6. Store object positions across frames
7. Compute displacement between consecutive frames
8. Estimate speed based on displacement and time
9. Display speed annotations on the video
10. Save the annotated output video

---

## 📏 Speed Calculation

Speed is estimated using the relationship between distance and time.
Basic formula:
```bash
speed = distance / time
```
In video analytics, distance is calculated from the pixel displacement between frames.
```bash
speed = pixel_distance / frame_time
```
Where:
- pixel_distance → displacement of the tracked object
- frame_time → time between frames (1 / FPS)

The estimated speed can be converted to real-world units such as km/h depending on calibration.

---

## 📊 Output

The generated output video contains:

- Bounding boxes around detected objects
- Unique tracking IDs
- Estimated speed values
- Annotated frames with speed labels
- Saved output video file

Example overlay:
```bash
#2 32 km/h
#5 27 km/h
```
---

## 🖥 Model Requirements

This example requires a YOLO ONNX object detection model.

You can:

- Export from Ultralytics YOLO
- Use an existing ONNX detection model
- Use your own trained detection model

Example export:

```bash
pip install ultralytics
yolo export model=yolov8x.pt format=onnx imgsz=1280
```

---

## 🎯 Practical Applications

- Highway speed monitoring
- Smart traffic management systems
- Autonomous driving datasets
- Vehicle behavior analytics
- Surveillance motion analysis
- Industrial vehicle tracking

---

## 📌 Notes

- GPU recommended for real-time performance
- Works best with fixed camera setups
- Speed estimation accuracy depends on camera calibration
- Multiple objects can be tracked simultaneously
- Stable video input improves tracking performance

---
