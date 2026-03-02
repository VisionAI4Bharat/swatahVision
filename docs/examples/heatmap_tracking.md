# 🔥 Heatmap & Multi-Object Tracking using YOLO (ONNX) + ByteTrack

This project demonstrates **crowd movement analysis** using a YOLO ONNX detection model and ByteTrack multi-object tracking.

It processes a video to:

- Detect people
- Track each person with a unique ID
- Generate a dynamic heatmap of movement
- Save an annotated output video

---

## 🚀 Features

* YOLO ONNX inference (GPU supported)
* Real-time multi-object tracking (ByteTrack)
* Person-only filtering (class_id = 0)
* Accumulated movement heatmap
* Configurable detection & tracking thresholds
* Annotated video export

---

## 📁 Project Structure

```
heatmap_and_track/
│
├── yolov11x-1280_onnx.py      # Main script
├── requirements.txt           # Dependencies
├── people_walking.mp4         # Sample input video
├── weight.onnx                # YOLO ONNX model (user provided)
└── README.md
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python numpy onnxruntime swatahvision
```

---

## ⬇️ Model Setup

### 1. Download YOLO ONNX Model

Download a YOLO ONNX model (example: `yolov11x-1280.onnx`).

Rename it to:

```
weight.onnx
```

Place it inside:

```
heatmap_and_track/
```

---

## ▶️ Run the Script

Basic run:

```bash
python yolov11x-1280_onnx.py \
    --source_weights_path weight.onnx
```

Run with custom video:

```bash
python yolov11x-1280_onnx.py \
    --source_weights_path weight.onnx \
    --source_video_path input_video.mp4 \
    --target_video_path output_video.mp4
```

---

## ⚙️ Command Line Arguments

### Required

`--source_weights_path`  
Path to YOLO ONNX weights file.

---

### Optional

`--source_video_path`  
Input video file.  
Default: `people_walking.mp4`

`--target_video_path`  
Output annotated video.  
Default: `output.mp4`

`--confidence_threshold`  
Detection confidence threshold.  
Default: `0.35`

`--iou_threshold`  
IoU threshold for NMS.  
Default: `0.5`

`--heatmap_alpha`  
Heatmap opacity (0–1).  
Default: `0.5`

`--radius`  
Radius of heat circle.  
Default: `25`

`--track_threshold`  
Confidence required to activate tracking.  
Default: `0.35`

`--track_seconds`  
Seconds to buffer lost tracks.  
Default: `5`

`--match_threshold`  
Threshold for matching tracks with detections.  
Default: `0.99`

---

## 🧠 How It Works

Pipeline:

1. Load YOLO ONNX model
2. Read video frames
3. Run object detection
4. Filter person class only
5. Apply ByteTrack multi-object tracking
6. Generate movement heatmap
7. Annotate frame with tracking IDs
8. Save output video

---

## 📊 Output

The output video contains:

- Accumulated crowd heatmap
- Persistent tracking IDs
- Person-only tracking
- Real-time visual annotations

---

## 🎯 Use Cases

- Crowd density analysis
- Retail footfall analytics
- Smart city monitoring
- Event crowd tracking
- Security surveillance

---

## 📌 Notes

- Currently tracks **person class only**
- GPU recommended for real-time performance
- Works best with stable camera footage
- Heatmap accumulates over time

---


