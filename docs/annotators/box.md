# Box Annotator – Drawing Bounding Boxes on Images

This code creates a **Box Annotator**, which is used to draw **bounding boxes** on an image based on object detection results.

In simple terms, it helps visualize where objects were detected in an image.

For example, if an AI model detects objects such as:

- cars
- people
- bicycles

the Box Annotator will **draw rectangles around those objects**.

---

# What is a Bounding Box?

A **bounding box** is a rectangle drawn around an object detected by an AI model.

Example:

```
+------------------+
|                  |
|       CAR        |
|                  |
+------------------+
```

The rectangle shows **where the object is located in the image**.

---

# Purpose of This Code

The main goal of this code is to:

1. Take an image
2. Take object detection results
3. Draw rectangles around detected objects
4. Return the annotated image

This helps **visualize AI predictions clearly**.

---

# Libraries Used

This code uses the following libraries:

- **NumPy** → handles numerical data
- **OpenCV (cv2)** → draws rectangles on images
- **SwatahVision modules** → manage detections and annotation tools

---

# Main Class

The main class created in this file is:

```
BoxAnnotator
```

This class is responsible for drawing bounding boxes on images.

---

# Configurable Settings

When creating a `BoxAnnotator`, you can configure a few options.

### 1. Color

Defines the color used to draw the bounding boxes.

Example:

```
color = ColorPalette.DEFAULT
```

Different objects can have different colors.

---

### 2. Thickness

Controls how thick the rectangle lines are.

Example:

```
thickness = 2
```

A higher value means **thicker box borders**.

---

### 3. Color Lookup Strategy

This decides **how colors are assigned to objects**.

Available strategies include:

- **INDEX** → color based on detection index
- **CLASS** → color based on object class
- **TRACK** → color based on object tracking ID

---

# Main Function: annotate()

The most important function in this class is:

```
annotate()
```

This function draws the bounding boxes.

### Inputs

The function takes three inputs:

#### 1. Scene

```
scene
```

The image where boxes will be drawn.

The image must be a **NumPy array**.

---

#### 2. Detections

```
detections
```

This contains information about detected objects.

Each detection includes the **coordinates of the bounding box**.

Example format:

```
x1, y1, x2, y2
```

These represent the corners of the rectangle.

---

#### 3. Custom Color Lookup (Optional)

```
custom_color_lookup
```

This allows users to manually control how colors are assigned.

If not provided, the default color strategy is used.

---

# How the Code Works

The function follows these steps:

### Step 1 – Loop Through Detections

The code processes each detected object one by one.

```
for detection_idx in range(len(detections))
```

---

### Step 2 – Get Bounding Box Coordinates

For each object, the code extracts the rectangle coordinates.

```
x1, y1, x2, y2
```

These represent the corners of the box.

---

### Step 3 – Choose a Color

The color for the bounding box is selected using the color lookup strategy.

This ensures consistent colors across detections.

---

### Step 4 – Draw the Rectangle

The rectangle is drawn using **OpenCV**:

```
cv2.rectangle()
```

The rectangle is placed on the image using the coordinates from the detection.

---

### Step 5 – Return the Annotated Image

After all boxes are drawn, the updated image is returned.

```
return scene
```

---

# Example Usage

Example of how this annotator might be used:

```python
import swatahvision as sv

image = ...
detections = sv.Detections(...)

box_annotator = sv.BoxAnnotator()

annotated_frame = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

This will return an image with **bounding boxes drawn around detected objects**.

---

# Workflow

The complete process looks like this:

```
Input Image
      ↓
Object Detection Model
      ↓
Detections (coordinates of objects)
      ↓
BoxAnnotator draws rectangles
      ↓
Annotated Image Output
```

---

# Why This Is Useful

Bounding box annotation is important because it helps:

- visualize AI model predictions
- debug detection models
- present results clearly
- understand where objects are located

---

# Summary

This code creates a tool that:

- receives detection results
- draws bounding boxes on images
- allows customizable colors and thickness
- returns an annotated image

It is commonly used in **computer vision applications such as object detection systems**.