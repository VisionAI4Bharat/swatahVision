# 📍 Time in Zone Analysis using YOLO (ONNX) + ByteTrack

This example demonstrates how to calculate **dwell time (time spent in a zone)** using computer vision.

It detects and tracks objects in a video and measures how long each tracked object remains inside predefined polygon zones.

This project is ideal for:

- 🛒 Retail analytics (checkout wait time monitoring)
- 🚦 Traffic management
- 🏭 Industrial safety zones
- 🛑 Restricted area monitoring
- 📊 Queue time analysis

---

## 🚀 What This Project Does

The system:

1. Detects objects using a YOLO ONNX model
2. Tracks objects using ByteTrack
3. Monitors predefined polygon zones
4. Calculates how long each object stays inside each zone
5. Displays dwell time in real-time
6. Saves the annotated output video

Each tracked object is labeled as:

```
#ID MM:SS
```

Example:

```
#5 01:24
```

Means object ID 5 stayed 1 minute and 24 seconds in the zone.

---

## 📁 Project Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/VisionAI4Bharat/swatahvision.git
cd swatahvision/examples/time_in_zone
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

## 🎨 Step 1: Draw Custom Zones

Before running dwell time analysis, you must define monitoring zones.

Use the provided script:

```bash
python scripts/draw_zones.py \
    --source_path "data/checkout.mp4" \
    --zone_configuration_path "data/config.json"
```

### Controls While Drawing

| Key | Action |
|------|--------|
| `Enter` | Finish current polygon |
| `Escape` | Cancel current polygon |
| `s` | Save zones to JSON file |
| `q` | Quit drawing window |

This generates a JSON file containing polygon coordinates.

Example `config.json`:

```json
[
    [[100,100], [400,100], [400,400], [100,400]]
]
```

Each polygon represents one monitoring zone.

---

## 🧠 Step 2: Run Time-in-Zone Analysis

Run the detection and tracking script:

```bash
python yolov11x-1280_onnx.py \
    --zone_configuration_path "data/config.json" \
    --source_video_path "data/checkout.mp4" \
    --source_weights_path "yolov11x-1280.onnx" \
    --classes 0 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```

---

## ⚙️ Command Line Arguments

### Required

`--zone_configuration_path`  
Path to zone configuration JSON file.

`--source_video_path`  
Path to input video file.

---

### Optional

`--source_weights_path`  
Path to YOLO ONNX model file.

`--classes`  
List of class IDs to track.  
Example:

```bash
--classes 0
```

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

1. Load YOLO ONNX model
2. Read video frames
3. Run object detection
4. Filter selected classes
5. Apply ByteTrack multi-object tracking
6. Check if object center lies inside polygon zone
7. Count frames inside zone
8. Convert frame count to seconds using FPS
9. Display timer on video
10. Save output video

---

## ⏱ Dwell Time Calculation

Time is calculated using video frame rate:

```
time_seconds = frame_count / FPS
```

The timer updates continuously while the object remains inside the zone.

If the object leaves the zone, timing stops.

---

## 📊 Output

The output video contains:

- Drawn polygon zones
- Tracked object IDs
- Real-time dwell time counter
- Color-coded zones
- Saved output video file

---

## 🖥 Model Requirements

This project requires a YOLO ONNX model.

You can:

- Export from Ultralytics YOLO
- Use a pre-exported ONNX model
- Use your internal trained model

Example export:

```bash
pip install ultralytics
yolo export model=yolov8x.pt format=onnx imgsz=1280
```

---

## 🎯 Practical Applications

- Checkout wait time monitoring
- Customer behavior analytics
- Smart traffic signal timing
- Industrial hazard zone tracking
- Security restricted-area alerts
- Queue performance monitoring

---

## 📌 Notes

- GPU recommended for real-time performance
- Zones are triggered using object center point
- Supports multiple zones
- Supports multiple object classes
- Works best with fixed camera footage

---

