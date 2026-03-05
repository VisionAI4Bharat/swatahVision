# Traffic Analysis using YOLO (ONNX) + ByteTrack

This example demonstrates how to perform traffic analysis from video streams using computer vision.

The system detects and tracks vehicles in a video and extracts useful traffic information such as vehicle counts, traffic flow, and movement patterns.

Traffic analysis systems are commonly used in smart city infrastructure, highway monitoring, and urban planning.

This project is ideal for:

- 🚗 Traffic monitoring systems
- 🚦 Smart traffic signal optimization
- 🏙 Smart city analytics
- 🛣 Highway traffic flow analysis
- 📊 Vehicle movement analytics

---

## 🚀 What This Project Does

The system performs the following steps:

1. Detects vehicles using a YOLO ONNX model
2. Tracks vehicles across frames using ByteTrack
3. Assigns unique IDs to each detected vehicle
4. Monitors vehicle movement across the video
5. Analyzes traffic flow patterns
6. Displays tracking information on the video
7. Saves the annotated output video

Each tracked vehicle is labeled as:

```
#ID
```
Example:

```
#7
```

This means the detected vehicle has tracking ID 7, which remains consistent while the vehicle is visible in the scene.

---

## 📁 Project Structure

```
traffic_analysis/
├── yolov11x-1280_onnx.py        # Main script for detection and traffic analysis
├── models/
│   └── yolov11x-1280.onnx       # YOLO ONNX detection model
├── data/
│   └── traffic.mp4              # Example traffic video
└── README.md                    # Example documentation
```

---

## 📁 Project Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/VisionAI4Bharat/swatahvision.git
cd swatahvision/examples/traffic_analysis
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

## ▶️ Run Traffic Analysis

Run the detection and tracking script:

```bash
python yolov11x-1280_onnx.py \
    --source_video_path "data/traffic.mp4" \
    --source_weights_path "models/yolov11x-1280.onnx" \
    --classes 2 5 7 \
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
List of class IDs to detect and track.  
Example:

```bash
--classes 2 5 7
```
Common COCO vehicle class IDs:
- 2 → Car
- 5 → Bus
- 7 → Truck

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

1. Load YOLO ONNX object detection model
2. Read frames from the input video
3. Perform object detection on each frame
4. Filter detections by selected vehicle classes
5. Apply ByteTrack multi-object tracking
6. Assign unique IDs to each vehicle
7. Track vehicle movements across frames
8. Analyze traffic flow and movement patterns
9. Draw bounding boxes and tracking IDs on frames
10. Save the annotated output video

Modern traffic analysis systems use object detection and tracking algorithms to monitor vehicle movement and traffic patterns in real-time video streams.

---

## 🚗 Traffic Metrics

Using object detection and tracking, the system can extract various traffic insights such as:

- Vehicle count in the scene
- Vehicle movement patterns
- Traffic density estimation
- Vehicle tracking trajectories

These metrics help analyze traffic congestion and flow patterns in monitored areas.

---

## 📊 Output

The generated output video contains:

- Bounding boxes around detected vehicles
- Unique tracking IDs
- Continuous vehicle tracking across frames
- Annotated video saved to disk

Example overlay:
```bash
#1
#3
#8
```
Each number represents a tracked vehicle ID.

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
YOLO models are widely used for real-time object detection tasks because they are fast and accurate for video analysis applications.
---

## 🎯 Practical Applications

- Smart traffic management systems
- Urban traffic monitoring
- Vehicle movement analysis
- Traffic congestion monitoring
- Intelligent transportation systems
---

## 📌 Notes

- GPU recommended for real-time performance
- Works best with fixed surveillance cameras
- Supports tracking multiple vehicles simultaneously
- Detection accuracy affects tracking quality
- Suitable for highway and urban traffic monitoring

---
