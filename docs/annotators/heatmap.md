# HeatMap Annotator – Visualizing Activity with Heatmaps

This code creates a **HeatMap Annotator**, which draws a **heatmap overlay** on an image based on object detections.

A heatmap shows **where objects appear most often** in a scene.  
The more frequently an object appears in a location, the **hotter (more intense)** that area becomes.

This helps visualize **activity patterns** over time.

---

# What is a Heatmap?

A **heatmap** is a visual representation of activity or intensity using colors.

Typical heatmap colors:

- 🔴 Red → High activity
- 🟡 Yellow → Medium activity
- 🔵 Blue → Low activity

Example:

```
Low activity        High activity
     🔵 → 🟡 → 🔴
```

If many detections happen in the same area, that area becomes **redder and brighter**.

---

# Purpose of This Code

This code is used to:

1. Track where objects appear in a scene
2. Accumulate those locations over time
3. Create a heatmap showing the most active areas
4. Overlay that heatmap on the original image

This is useful for **analyzing movement patterns**.

---

# Libraries Used

The code uses the following libraries:

- **NumPy** → handles numerical data
- **OpenCV (cv2)** → image processing and drawing
- **SwatahVision modules** → detections and annotation utilities

---

# Main Class

The main class created in this file is:

```
HeatMapAnnotator
```

This class is responsible for creating and updating a heatmap on images.

---

# Configurable Settings

When creating a `HeatMapAnnotator`, several parameters can be adjusted.

---

## 1. Position

```
position = Position.BOTTOM_CENTER
```

This determines **where the heat point is placed on each detected object**.

Examples:

- center of object
- bottom center
- top center

---

## 2. Opacity

```
opacity = 0.2
```

Controls how transparent the heatmap overlay is.

Range:

```
0 → invisible
1 → fully visible
```

---

## 3. Radius

```
radius = 40
```

This defines the **size of the heat circle** drawn for each detection.

Larger radius = larger heat spots.

---

## 4. Kernel Size

```
kernel_size = 25
```

This controls how much the heatmap is **blurred**.

Blurring helps create a smooth heatmap instead of sharp circles.

---

## 5. Heatmap Colors

Two values control the heatmap color range:

```
top_hue = 0
low_hue = 125
```

These represent color values in **HSV color format**.

Typical mapping:

```
Red   → High activity
Blue  → Low activity
```

---

# Main Function: annotate()

The main function used to draw the heatmap is:

```
annotate()
```

This function takes:

### 1. Scene

```
scene
```

The image where the heatmap will be drawn.

It must be an image in **NumPy format**.

---

### 2. Detections

```
detections
```

This contains the detected objects in the image.

Each detection provides the **location of an object**.

---

# How the Heatmap Works

The heatmap is built step-by-step.

---

## Step 1 – Create Heat Mask

The code creates a **heat mask**, which stores activity levels for each pixel.

Initially it contains zeros.

```
self.heat_mask = np.zeros(...)
```

---

## Step 2 – Mark Detection Points

For each detected object:

- a circle is drawn
- the circle represents activity

Example:

```
cv2.circle()
```

Each detection adds **heat to that location**.

---

## Step 3 – Accumulate Heat Over Time

The heatmap is **not reset each frame**.

Instead, it keeps adding new detections:

```
self.heat_mask = mask + self.heat_mask
```

This allows the heatmap to show **long-term activity patterns**.

---

## Step 4 – Convert Heat to Colors

The heat values are converted into colors using the HSV color system.

This produces colors like:

```
Blue → Low activity
Yellow → Medium activity
Red → High activity
```

---

## Step 5 – Smooth the Heatmap

The heatmap is blurred using:

```
cv2.blur()
```

This creates smooth gradients between hot areas.

---

## Step 6 – Overlay Heatmap on Image

Finally, the heatmap is blended with the original image using:

```
cv2.addWeighted()
```

This creates a **semi-transparent overlay**.

---

# Example Usage

Example usage in a video processing pipeline:

```python
import swatahvision as sv

heat_map_annotator = sv.HeatMapAnnotator()

detections = ...

annotated_frame = heat_map_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

This will produce an image with a **heatmap overlay showing activity locations**.

---

# Workflow

The full workflow looks like this:

```
Input Frame
      ↓
Object Detection Model
      ↓
Detections (object positions)
      ↓
HeatMapAnnotator accumulates heat
      ↓
Heatmap overlay on image
```

---

# Why Heatmaps Are Useful

Heatmaps help analyze **behavior and movement patterns**.

Common applications include:

- retail store analytics
- crowd movement analysis
- sports tracking
- traffic monitoring
- security surveillance

They show **where activity happens most frequently**.

---

# Summary

This code creates a heatmap visualization tool that:

- tracks object locations
- accumulates activity over time
- converts activity into color intensity
- overlays the heatmap on the original image

It is commonly used in **computer vision analytics systems**.