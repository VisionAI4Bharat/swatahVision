# ByteTrack Tracker

## Overview

**ByteTrack** is a high-performance multi-object tracking algorithm used to track objects across video frames.

It works by associating **detections from object detection models** with existing tracks and assigning a **unique tracker ID** to each object.

ByteTrack is widely used in:

- Object tracking
- Surveillance systems
- Traffic monitoring
- Sports analytics
- Crowd analysis

The tracker works seamlessly with **SwatahVision Detections**.

---

# Class: `ByteTrack`

```python
class ByteTrack
```

The `ByteTrack` class performs **multi-object tracking** using bounding box detections.

It assigns **persistent IDs to objects across frames**.

---

# Constructor

```python
ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=1
)
```

---

# Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `track_activation_threshold` | float | Minimum detection confidence required to activate a track |
| `lost_track_buffer` | int | Number of frames a track is kept after it disappears |
| `minimum_matching_threshold` | float | IOU threshold used when matching tracks with detections |
| `frame_rate` | int | Frame rate of the processed video |
| `minimum_consecutive_frames` | int | Number of frames an object must appear before being considered a valid track |

---

# How ByteTrack Works

ByteTrack tracks objects using the following pipeline:

1. **Object Detection**

Objects are detected using models such as:

- YOLO
- SSD
- RetinaNet

2. **Bounding Box Association**

Detected boxes are matched with existing tracks using **IoU distance**.

3. **Kalman Filter Prediction**

Object motion is predicted using a **Kalman Filter**.

4. **Track Management**

Tracks are categorized as:

- **Tracked** – currently active objects
- **Lost** – temporarily missing objects
- **Removed** – expired tracks

---

# Method: `update_with_detections`

```python
update_with_detections(detections: Detections) -> Detections
```

Updates the tracker using detection results and returns updated detections with tracker IDs.

---

## Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `detections` | `Detections` | Detection results containing bounding boxes and confidence scores |

---

## Returns

```
Detections
```

Updated detections with **tracker_id assigned**.

---

# Example Usage

```python
import swatahvision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def callback(frame, index):

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    detections = tracker.update_with_detections(detections)

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

    frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    frame = label_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

    return frame
```

---

# Reset Tracker

```python
tracker.reset()
```

Resets the internal state of the tracker.

This clears:

- tracked objects
- lost objects
- removed objects

Useful when processing **multiple videos sequentially**.

---

# Internal Tracking Pipeline

ByteTrack internally performs several steps:

### 1. Detection Filtering

Detections are separated based on confidence thresholds.

### 2. First Association

High confidence detections are matched with existing tracks.

### 3. Second Association

Lower confidence detections are matched to recover lost tracks.

### 4. Track Initialization

New tracks are created for unmatched detections.

### 5. Track Removal

Tracks that remain lost for too long are removed.

---

# Track States

Tracks can exist in three states:

| State | Description |
|------|-------------|
| `Tracked` | Object is currently visible |
| `Lost` | Object temporarily disappeared |
| `Removed` | Object removed after long absence |

---

# Helper Functions

ByteTrack uses several internal helper functions.

---

## `joint_tracks`

```python
joint_tracks(track_list_a, track_list_b)
```

Combines two track lists while avoiding duplicate tracks.

---

## `sub_tracks`

```python
sub_tracks(track_list_a, track_list_b)
```

Removes tracks from one list that exist in another list.

---

## `remove_duplicate_tracks`

```python
remove_duplicate_tracks(tracks_a, tracks_b)
```

Removes duplicate tracks based on **IoU similarity**.

---

# Dependencies

ByteTrack relies on the following modules:

- NumPy
- Kalman Filter
- IoU matching utilities
- SwatahVision Detection module

---

# Example Visualization

When combined with annotators like:

- `BoxAnnotator`
- `LabelAnnotator`
- `TraceAnnotator`

you can visualize object tracking across frames.

Example output:

```
Person #1
Person #2
Car #3
```

Each object keeps the **same ID across frames**.

---

# Summary

The **ByteTrack tracker** enables reliable multi-object tracking in video streams.

Key features:

- High accuracy tracking
- Kalman filter motion prediction
- Robust handling of occlusions
- Persistent object IDs
- Easy integration with detection models

It is an essential component for building **real-time computer vision tracking systems**.

---

## 👨‍💻 Author

- **Atharva Kotkar** 