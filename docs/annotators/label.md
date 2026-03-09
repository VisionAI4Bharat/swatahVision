# Label Annotator

## Overview

The **LabelAnnotator** is used to display text labels on detected objects in an image or video frame.  
It places a **text label with a background box** near each detection bounding box.

Labels typically contain information such as:

- Object class name
- Confidence score
- Tracker ID
- Custom text

This annotator is commonly used in **object detection and tracking visualization pipelines**.

---

# Class: `LabelAnnotator`

The `LabelAnnotator` class extends `_BaseLabelAnnotator` and provides functionality to draw labels on detected objects.

```python
class LabelAnnotator(_BaseLabelAnnotator)
```

It supports customizable:

- Text appearance
- Background color
- Label positioning
- Smart overlap avoidance
- Multi-line labels

---

# Constructor

```python
LabelAnnotator(
    color=ColorPalette.DEFAULT,
    color_lookup=ColorLookup.CLASS,
    text_color=Color.WHITE,
    text_scale=0.5,
    text_thickness=1,
    text_padding=10,
    text_position=Position.TOP_LEFT,
    text_offset=(0,0),
    border_radius=0,
    smart_position=False,
    max_line_length=None
)
```

---

# Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `color` | `Color \| ColorPalette` | Background color for label box |
| `color_lookup` | `ColorLookup` | Strategy for mapping colors |
| `text_color` | `Color \| ColorPalette` | Color used for label text |
| `text_scale` | `float` | Font size scale |
| `text_thickness` | `int` | Thickness of text characters |
| `text_padding` | `int` | Padding around text inside label box |
| `text_position` | `Position` | Position of label relative to bounding box |
| `text_offset` | `(int, int)` | Offset applied to label position |
| `border_radius` | `int` | Radius for rounded label box corners |
| `smart_position` | `bool` | Automatically reposition labels to avoid overlap |
| `max_line_length` | `int \| None` | Maximum characters per line before wrapping |

---

# Color Lookup Strategies

Label colors can be determined using different strategies.

| Option | Description |
|------|-------------|
| `INDEX` | Color determined by detection index |
| `CLASS` | Color determined by object class |
| `TRACK` | Color determined by tracker ID |

---

# Method: `annotate`

```python
annotate(scene, detections, labels=None, custom_color_lookup=None)
```

Adds labels to detected objects in the provided image.

---

## Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `scene` | `ImageType` | Input image (`numpy.ndarray` or `PIL.Image`) |
| `detections` | `Detections` | Detection results containing bounding boxes |
| `labels` | `list[str]` | Custom label text for each detection |
| `custom_color_lookup` | `np.ndarray` | Optional custom color mapping |

---

## Returns

Annotated image containing labels.

```
numpy.ndarray or PIL.Image.Image
```

---

# How Label Annotation Works

1. The detection bounding boxes are obtained.
2. Anchor coordinates are calculated based on `text_position`.
3. Label text size is computed using OpenCV.
4. A background box is created for the label.
5. The label text is drawn on the image.

---

# Smart Positioning

When `smart_position=True`, the annotator attempts to **prevent label overlap** by spreading label boxes.

It uses internal utilities such as:

```
spread_out_boxes
snap_boxes
```

This ensures labels remain readable even when many detections appear close together.

---

# Multi-Line Labels

Long labels can be automatically wrapped.

Example:

```
Person
Confidence: 0.92
```

This is controlled using:

```
max_line_length
```

---

# Example Usage

```python
import swatahvision as sv

image = ...
detections = sv.Detections(...)

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

label_annotator = sv.LabelAnnotator(
    text_position=sv.Position.CENTER
)

annotated_frame = label_annotator.annotate(
    scene=image.copy(),
    detections=detections,
    labels=labels
)
```

---

# Example Output

The annotator draws a **text label with a colored background**.

Example visualization:

```
┌─────────────┐
│ Person 0.92 │
└─────────────┘
```

Positioned near the detection bounding box.

---

# Rounded Label Boxes

If `border_radius > 0`, the background label box will have rounded edges.

Example:

```
LabelAnnotator(border_radius=10)
```

---

# Error Handling

### Invalid Label Count

If the number of labels does not match the number of detections:

```
ValueError
```

This prevents mismatched annotations.

---

# Dependencies

The LabelAnnotator relies on:

- OpenCV (`cv2`)
- NumPy
- SwatahVision geometry utilities
- SwatahVision detection modules

---

# Summary

The **LabelAnnotator** provides a flexible way to add text labels to detection results.

Key features:

- Customizable label appearance
- Multi-line text support
- Smart label positioning
- Color mapping strategies
- Rounded label backgrounds

It is commonly used together with:

- `BoxAnnotator`
- `TraceAnnotator`
- `HeatMapAnnotator`

to build complete **visualization pipelines for computer vision models**.

---

## 👨‍💻 Author

- **Atharva Kotkar**  