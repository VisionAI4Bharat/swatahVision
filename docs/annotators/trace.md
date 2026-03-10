# Trace Annotator

## Overview

The **TraceAnnotator** is used to visualize the movement path of tracked objects across video frames.  
It draws **trace lines** showing where an object has traveled based on its **tracker ID**.

This annotator is useful for:

- Object tracking visualization
- Motion analysis
- Sports analytics
- Surveillance systems
- Traffic monitoring

The trace is generated using the **historical positions of tracked detections**.

---

# Class: `TraceAnnotator`

The `TraceAnnotator` class extends `BaseAnnotator` and is responsible for drawing trace paths for tracked objects.

```python
class TraceAnnotator(BaseAnnotator)
```

It uses **tracker IDs** from the `Detections` object to associate positions across frames.

---

# Requirements

This annotator **requires object tracking**.

The `Detections` object must contain:

```
detections.tracker_id
```

If `tracker_id` is missing, the annotator will raise an error.

---

# Constructor

```python
TraceAnnotator(
    color=ColorPalette.DEFAULT,
    position=Position.CENTER,
    trace_length=30,
    thickness=2,
    smooth=False,
    color_lookup=ColorLookup.CLASS
)
```

## Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `color` | `Color \| ColorPalette` | Color used for the trace line |
| `position` | `Position` | Anchor position used for trace calculation |
| `trace_length` | `int` | Maximum number of historical points stored for each object |
| `thickness` | `int` | Thickness of the trace line |
| `smooth` | `bool` | Enables spline smoothing for the trace |
| `color_lookup` | `ColorLookup` | Strategy for assigning colors |

---

# Color Lookup Options

| Option | Description |
|------|-------------|
| `INDEX` | Color determined by detection index |
| `CLASS` | Color determined by object class |
| `TRACK` | Color determined by tracker ID |

---

# Method: `annotate`

```python
annotate(scene, detections, custom_color_lookup=None)
```

Draws the trace path on the given frame.

## Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `scene` | `ImageType` | Input image (`numpy.ndarray` or `PIL.Image`) |
| `detections` | `Detections` | Detection results containing bounding boxes and tracker IDs |
| `custom_color_lookup` | `np.ndarray` | Optional custom color mapping |

## Returns

Annotated image with trace lines.

```
numpy.ndarray or PIL.Image.Image
```

---

# How Trace Works

1. Object detections are received from the model.
2. Each detection is associated with a **tracker ID**.
3. Historical positions for each tracker ID are stored.
4. Lines are drawn between stored positions to create a trace path.

---

# Trace Smoothing

If `smooth=True`, the trace path is smoothed using **B-spline interpolation** from `scipy`.

```python
splprep
splev
```

This produces smoother curves instead of straight line segments.

---

# Example Usage

```python
import swatahvision as sv
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

trace_annotator = sv.TraceAnnotator()

tracker = sv.ByteTrack()

video_info = sv.VideoInfo.from_video_path(video_path="input.mp4")

frames_generator = sv.get_video_frames_generator(source_path="input.mp4")

with sv.VideoSink(target_path="output.mp4", video_info=video_info) as sink:

    for frame in frames_generator:

        result = model(frame)[0]

        detections = sv.Detections.from_ultralytics(result)

        detections = tracker.update_with_detections(detections)

        annotated_frame = trace_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        sink.write_frame(frame=annotated_frame)
```

---

# Example Output

The annotator will draw a line showing the **movement path of each tracked object** across frames.

Example visualization:

```
Object Path
   ●────●────●────●
```

Each object keeps its own trace based on its **tracker ID**.

---

# Error Handling

### Missing Tracker ID

If the `Detections` object does not contain tracker IDs:

```
ValueError: The tracker_id field is missing in the provided detections
```

Tracking must be enabled before using this annotator.

---

# Dependencies

The TraceAnnotator relies on the following modules:

- OpenCV (`cv2`)
- NumPy
- SciPy
- SwatahVision detection and drawing modules

---

# Summary

The **TraceAnnotator** provides an easy way to visualize object movement over time.

Key features:

- Draws historical object paths
- Works with tracking pipelines
- Supports color customization
- Optional smooth curves
- Compatible with video processing pipelines

This annotator is particularly useful for applications involving **object tracking and motion visualization**.

---

## 👨‍💻 Author

- **Atharva Kotkar**    