# Object Tracking using YOLO (ONNX) + ByteTrack

This example demonstrates how to perform multi-object tracking in a video using computer vision.

The system detects objects in each frame and assigns a consistent tracking ID to each object as it moves across frames.

Object tracking allows systems to understand movement patterns and trajectories of objects in video streams.

This is ideal for:

- 🚗 Traffic monitoring systems
- 🏬 Retail customer movement analysis
- 🏭 Industrial object tracking
- 🎥 Smart surveillance systems
- 📊 Motion analytics in video streams

---

## 🚀 What This Project Does

The system performs the following steps:

1. Detects objects using a YOLO ONNX model
2. Tracks objects across frames using ByteTrack
3. Assigns a unique ID to each detected object
4. Maintains object identity across frames
5. Draws bounding boxes and tracking IDs on the video
6. Saves the annotated output video

Each detected object is labeled as:

```
#ID
```
Example:

```
#3
```

This means the detected object has tracking ID 3 and will keep the same ID as long as it remains visible.

---

## 📁 Project Structure

```
tracking/
├── yolov11x-1280_onnx.py        # Main tracking script
├── models/
│   └── yolov11x-1280.onnx       # YOLO ONNX detection model
├── data/
│   └── input.mp4                # Example input video
└── README.md                    # Example documentation
```

---

## 📁 Project Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/VisionAI4Bharat/swatahvision.git
cd swatahvision/examples/tracking
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

## 🚗 Run Object Tracking

Run the tracking script:

```bash
python yolov11x-1280_onnx.py \
    --source_video_path "data/input.mp4" \
    --source_weights_path "models/yolov11x-1280.onnx" \
    --classes 0 \
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
Path to YOLO ONNX detection model .

`--classes`  
List of class IDs to track.  
Example:

```bash
--classes 0
```
Common class IDs:
- 0 → Person
- 2 → Car
- 5 → Bus

If this argument is empty, all detected classes will be tracked.

`--confidence_threshold`
Detection confidence threshold.  
Default: `0.3`

`--iou_threshold`  
IoU threshold for Non-Max Suppression.  
Default: `0.7`

---

## 🧠 How It Works (Technical Overview)

Processing Pipeline:

1. Load YOLO ONNX object detection model
2. Read frames from the input video
3. Run object detection on each frame
4. Filter detections by selected classes
5. Pass detections to the ByteTrack tracker
6. Assign unique tracking IDs to objects
7. Maintain identity across frames
8. Draw bounding boxes and IDs on the video
9. Save annotated output video

Multi-object tracking systems aim to estimate object locations and maintain consistent identities across frames in a video.

---

## 🧠 ByteTrack Tracking Algorithm

This example uses ByteTrack, a high-performance multi-object tracking algorithm.

Key features:

- Tracks multiple objects simultaneously
- Maintains consistent IDs across frames
- Handles occlusions and missed detections
- Works in real-time video analytics systems
ByteTrack associates detection boxes across frames to maintain object identities.

---

## 📊 Output

The generated output video contains:
- Bounding boxes around detected objects
- Unique tracking IDs for each object
- Continuous tracking across frames
- Annotated output video saved to disk

Example overlay:
```bash
#1
#4
#7
```
Each number represents a tracked object ID.

---

## 🖥 Model Requirements

This example requires a YOLO ONNX object detection model.

You can:

- Export a model from Ultralytics YOLO
- Use a pre-trained ONNX model
- Use your own trained detection model

Example export:

```bash
pip install ultralytics
yolo export model=yolov8x.pt format=onnx imgsz=1280
```

---

## 🎯 Practical Applications

- Smart surveillance systems
- Crowd movement analysis
- Traffic monitoring and vehicle tracking
- Industrial object monitoring
- Retail customer behavior analysis

---

## 📌 Notes

- GPU recommended for real-time performance
- Works best with stable camera footage
- Supports tracking multiple objects simultaneously
- Tracking accuracy depends on detection quality
- Suitable for both indoor and outdoor video analytics

---
