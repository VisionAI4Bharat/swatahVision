# 📍 Count People in Zone using YOLO (ONNX) + ByteTrack

This example demonstrates how to count the number of people inside predefined polygon zones using computer vision.

It detects and tracks people in a video and determines how many individuals are present inside each defined zone in real-time.

This project is ideal for:

- 🏬 Retail store occupancy monitoring
- 🚪 Entry/exit people counting
- 🏫 Classroom or hall occupancy tracking
- 🚧 Restricted area monitoring
- 🧑‍🤝‍🧑 Crowd density estimation

---

## 🚀 What This Project Does

The system:

1. Detects people using a YOLO ONNX model
2. Tracks individuals using ByteTrack
3. Monitors predefined polygon zones
4. Checks whether a tracked person is inside a zone
5. Counts the number of people inside each zone
6. Displays the count in real-time
7. Saves the annotated output video
Each zone displays the number of people currently inside it.

Example:

```
Zone 1: 5
Zone 2: 2
```

Means:

```
5 people are inside Zone 1
2 people are inside Zone 2
```
The count updates dynamically as people enter or leave the zone.

---

## 📥 Model Download

Pretrained models for **swatahVision** are available in the Model Zoo.

🔗 [https://visionai4bharat.github.io/swatahVision/model_zoo/](https://visionai4bharat.github.io/swatahVision/model_zoo/)

---

## 📁 Project Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/VisionAI4Bharat/swatahVision.git
cd swatahVision/examples/count_people_in_zone
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

Before running people counting, you must define monitoring zones.

Use the provided script:

```bash
python scripts/draw_zones.py \
    --source_path "data/input.mp4" \
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

## 🧠 Step 2: Run People Counting

Run the detection and tracking script:

```bash
python yolov11x-1280_onnx.py \
    --zone_configuration_path "data/config.json" \
    --source_video_path "data/input.mp4" \
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
Class 0 corresponds to person detection in most YOLO models.
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
4. Filter selected classes (typically person)
5. Apply ByteTrack multi-object tracking
6. Compute the center point of each tracked object
7. Check if object center lies inside polygon zone
8. Count number of people inside each zone
9. Display counts on the video
10. Save the output video

---

## 👥 People Counting Logic

People counting is performed by checking whether the center point of a tracked bounding box lies inside a polygon zone.

If the center point lies inside the zone, the person is counted as present in that zone.

The count updates continuously for each frame.

If a person leaves the zone, the count decreases automatically.

---

## 📊 Output

The output video contains:

- Drawn polygon zones
- Tracked object IDs
- Real-time people count per zone
- Bounding boxes around detected people
- Saved annotated output video

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

- Retail store occupancy monitoring
- Crowd density analysis
- Smart building management
- Security restricted-area monitoring
- Event crowd management
- Smart city pedestrian analytics
---

## 📌 Notes

- GPU recommended for real-time performance
- Zones are triggered using object center point
- Supports multiple zones
- Supports multiple object classes
- Works best with fixed camera footage
---
