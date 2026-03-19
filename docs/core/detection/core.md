# Detections Core API

This document explains how the **Detections class** in **swatahVision** works.

The `Detections` class is the **core data structure used for object detection results**.  
It stores information about detected objects such as:

- Bounding boxes
- Confidence scores
- Class IDs
- Segmentation masks
- Tracking IDs
- Additional metadata

It also provides utilities for:

- Filtering detections
- Non-Maximum Suppression (NMS)
- Non-Maximum Merging (NMM)
- Merging detections from multiple models
- Accessing anchor positions of objects

---

# Installation

Install the required libraries.

```bash
pip install swatahVision
pip install numpy
pip install opencv-python
```

---

# Import Libraries

```python
import swatahVision as sv
import numpy as np
```

---

# What is the `Detections` Class?

The `Detections` class stores information about detected objects.

Each detection may contain:

| Field | Description |
|-----|-----|
| `xyxy` | Bounding box coordinates `[x1, y1, x2, y2]` |
| `confidence` | Detection confidence score |
| `class_id` | Predicted object class |
| `mask` | Segmentation mask (optional) |
| `tracker_id` | Object tracking ID (optional) |
| `data` | Additional user data |
| `metadata` | Extra information |

---

# Creating a Detections Object

Example:

```python
import numpy as np
import swatahVision as sv

detections = sv.Detections(
    xyxy=np.array([[50, 60, 200, 220]]),
    confidence=np.array([0.89]),
    class_id=np.array([1])
)

print(detections)
```

---

# Creating Detections from Model Outputs

swatahVision provides helper methods to convert model outputs.

## YOLO

```python
detections = sv.Detections.from_yolo(outputs, conf_threshold=0.5)
```

This converts YOLO predictions into a `Detections` object.

Parameters:

| Parameter | Description |
|---|---|
| `outputs` | Raw model output |
| `conf_threshold` | Minimum confidence score |
| `nms_threshold` | Optional NMS threshold |
| `class_agnostic` | Ignore class IDs during NMS |

---

## SSD

```python
detections = sv.Detections.from_ssd(outputs, conf_threshold=0.5)
```

This converts SSD predictions into structured detections.

---

## RetinaNet

```python
detections = sv.Detections.from_retinanet(outputs, conf_threshold=0.5)
```

This converts RetinaNet predictions into `Detections`.

---

# Iterating Through Detections

You can loop through all detections.

```python
for xyxy, mask, confidence, class_id, tracker_id, data in detections:
    print("Box:", xyxy)
    print("Confidence:", confidence)
    print("Class:", class_id)
```

Example output:

```
Box: [45 60 210 220]
Confidence: 0.91
Class: 2
```

---

# Get Number of Detections

```python
print(len(detections))
```

Example output:

```
3
```

---

# Access Specific Detections

Get the first detection:

```python
first = detections[0]
```

Get first 10 detections:

```python
subset = detections[0:10]
```

Filter detections:

```python
high_conf = detections[detections.confidence > 0.7]
```

Filter by class:

```python
persons = detections[detections.class_id == 0]
```

---

# Adding Custom Data

You can attach additional information.

Example:

```python
detections["names"] = ["car", "person", "bicycle"]
```

This stores extra metadata inside the detection object.

---

# Checking Empty Detections

Create empty detections:

```python
detections = sv.Detections.empty()
```

Check if empty:

```python
detections.is_empty()
```

---

# Merging Detections

You can combine multiple detection sets.

Example:

```python
merged = sv.Detections.merge([detections1, detections2])
```

This is useful when using **multiple detection models**.

Example result:

```
Original model 1 detections : 3
Original model 2 detections : 2

Merged detections : 5
```

---

# Non-Maximum Suppression (NMS)

Object detection models may detect the same object multiple times.

NMS removes duplicate detections.

```python
filtered = detections.with_nms(threshold=0.5)
```

Parameters:

| Parameter | Description |
|---|---|
| `threshold` | IoU threshold |
| `class_agnostic` | Ignore class IDs |
| `overlap_metric` | Overlap metric (IOU or IOS) |

---

# Non-Maximum Merging (NMM)

NMM merges overlapping detections instead of removing them.

```python
merged = detections.with_nmm(threshold=0.5)
```

This combines multiple overlapping boxes into one larger detection.

---

# Getting Anchor Coordinates

You can get anchor positions inside bounding boxes.

Example:

```python
from swatahVision.geometry.core import Position

centers = detections.get_anchors_coordinates(Position.CENTER)
print(centers)
```

Supported anchor positions:

| Anchor | Description |
|---|---|
| `CENTER` | center of box |
| `TOP_LEFT` | top left corner |
| `TOP_RIGHT` | top right corner |
| `BOTTOM_LEFT` | bottom left corner |
| `BOTTOM_RIGHT` | bottom right corner |
| `CENTER_LEFT` | left center |
| `CENTER_RIGHT` | right center |
| `BOTTOM_CENTER` | bottom center |

---

# Example Full Workflow

```python
import swatahVision as sv

model = sv.Model(
    model="yolov8n.xml",
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)

image = sv.Image.load_from_file("image.jpg")

outputs = model(image)

detections = sv.Detections.from_yolo(outputs, conf_threshold=0.5)

detections = detections.with_nms(threshold=0.5)

for box, mask, confidence, class_id, tracker_id, data in detections:
    print("Object:", class_id, "Confidence:", confidence)
```

Example output:

```
Object: 2 Confidence: 0.91
Object: 0 Confidence: 0.87
```

---

# Summary

The `Detections` class is the **core structure used by swatahVision for object detection results**.

It helps developers to:

- Store detection results
- Convert model outputs
- Filter detections
- Remove duplicate detections
- Merge results from multiple models
- Access bounding box positions

This makes it easier to build **computer vision pipelines for real-world applications**.

---