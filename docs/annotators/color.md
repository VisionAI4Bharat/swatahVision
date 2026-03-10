# Color Annotator – Highlighting Detected Objects with Color Masks

This code creates a **Color Annotator**, which highlights detected objects in an image using **colored overlays**.

Instead of just drawing a rectangle outline around an object, this annotator **fills the bounding box with a semi-transparent color**.

This helps make detected objects **more visible and easier to understand**.

---

# What This Annotator Does

The Color Annotator:

1. Takes an image
2. Takes object detection results
3. Draws colored rectangles over detected objects
4. Blends those rectangles with the original image

The result is an image where detected objects appear **highlighted with colored masks**.

---

# Example Idea

Original image:

```
Person walking near a car
```

After annotation:

```
Person highlighted with a color mask
Car highlighted with another color mask
```

The colored regions help visually identify detected objects.

---

# Libraries Used

This code uses the following libraries:

- **NumPy** → handles numerical data
- **OpenCV (cv2)** → draws rectangles and blends images
- **SwatahVision modules** → handle detections and annotation utilities

---

# Main Class

The main class created in this file is:

```
ColorAnnotator
```

This class is responsible for drawing **colored masks over detected objects**.

---

# Configurable Settings

When creating a `ColorAnnotator`, several settings can be adjusted.

---

## 1. Color

```
color = ColorPalette.DEFAULT
```

Defines the colors used for object masks.

The annotator can use:

- a single color
- a palette of colors for different objects

---

## 2. Opacity

```
opacity = 0.5
```

Controls how transparent the colored mask is.

Range:

```
0 → invisible
1 → fully solid color
```

Example:

```
0.5 → semi-transparent mask
```

---

## 3. Color Lookup Strategy

```
color_lookup = ColorLookup.CLASS
```

This determines **how colors are assigned to objects**.

Possible options:

- **INDEX** → color based on detection order
- **CLASS** → color based on object type
- **TRACK** → color based on tracking ID

---

# Main Function: annotate()

The key function in this class is:

```
annotate()
```

This function draws colored masks on detected objects.

---

# Inputs

The function takes three inputs.

---

## 1. Scene

```
scene
```

The image where annotations will be drawn.

The image must be a **NumPy array**.

---

## 2. Detections

```
detections
```

This contains information about detected objects.

Each detection includes the **coordinates of the bounding box**.

Example format:

```
x1, y1, x2, y2
```

These coordinates define the rectangle around an object.

---

## 3. Custom Color Lookup (Optional)

```
custom_color_lookup
```

This allows the user to manually control how colors are assigned.

If not provided, the default color strategy is used.

---

# How the Code Works

The annotation process happens in several steps.

---

## Step 1 – Copy the Original Image

A copy of the image is created.

```
scene_with_boxes = scene.copy()
```

This allows drawing masks without modifying the original image immediately.

---

## Step 2 – Loop Through Each Detection

The code processes every detected object.

```
for detection_idx in range(len(detections))
```

---

## Step 3 – Get Bounding Box Coordinates

For each detection, the code extracts the rectangle coordinates:

```
x1, y1, x2, y2
```

These represent the corners of the object box.

---

## Step 4 – Choose the Color

The code selects a color based on the chosen color strategy.

Different objects may have different colors.

---

## Step 5 – Draw the Colored Mask

A filled rectangle is drawn using OpenCV:

```
cv2.rectangle()
```

The rectangle covers the detected object.

---

## Step 6 – Blend with the Original Image

The colored masks are blended with the original image using:

```
cv2.addWeighted()
```

This creates a **semi-transparent overlay**.

---

# Example Usage

Example of how this annotator can be used:

```python
import swatahvision as sv

image = ...
detections = sv.Detections(...)

color_annotator = sv.ColorAnnotator()

annotated_frame = color_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

The output will be an image where detected objects are **highlighted with colored masks**.

---

# Workflow

The overall process looks like this:

```
Input Image
      ↓
Object Detection Model
      ↓
Detections (object coordinates)
      ↓
ColorAnnotator adds colored masks
      ↓
Annotated Image Output
```

---

# Why This Is Useful

Color masks help make detection results **easier to understand visually**.

They are useful for:

- object detection visualization
- video analytics
- surveillance systems
- autonomous driving research
- AI demonstrations

---

# Summary

This code creates a tool that:

- receives detection results
- draws colored masks over detected objects
- blends the masks with the original image
- produces a clear visual representation of detected objects

It is commonly used in **computer vision systems to highlight detected objects clearly**.