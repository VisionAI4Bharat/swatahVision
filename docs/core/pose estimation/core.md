# Pose Data Structure and MoveNet Parser

This document explains how the **Pose class** works.

The `Pose` class is a simple data structure used to represent **human pose estimation results**.

It stores:

- Keypoint coordinates
- Confidence scores for each keypoint

It also provides utilities for:

- Iterating through keypoints
- Getting the number of keypoints
- Converting **MoveNet model outputs** into a structured pose format

---

# Installation

Install required libraries.

```bash
pip install numpy
```

---

# Import Libraries

```python
import numpy as np
from pose import Pose
```

---

# What is the `Pose` Class?

The `Pose` class stores **pose estimation results**.

Pose estimation models detect important **body joints**, called **keypoints**.

Examples include:

- Nose
- Eyes
- Shoulders
- Elbows
- Wrists
- Hips
- Knees
- Ankles

Each keypoint contains:

| Field | Description |
|------|------|
| `keypoints` | Coordinates `[x, y]` of the keypoint |
| `confidence` | Confidence score for that keypoint |

---

# Pose Data Structure

The Pose object contains two arrays.

| Attribute | Shape | Description |
|------|------|------|
| `keypoints` | `(K,2)` | Keypoint coordinates `(x,y)` |
| `confidence` | `(K,)` | Confidence score for each keypoint |

Example:

```
keypoints =
[[320, 240],
 [330, 260],
 [340, 280],
 ...
]

confidence =
[0.92, 0.88, 0.95, ...]
```

---

# Creating a Pose Object

Example:

```python
import numpy as np

pose = Pose(
    keypoints=np.array([[100, 200], [120, 220]]),
    confidence=np.array([0.9, 0.85])
)

print(pose)
```

---

# Getting the Number of Keypoints

The `__len__()` method returns the number of keypoints.

```python
print(len(pose))
```

Example output:

```
17
```

Most pose estimation models detect **17 keypoints**.

---

# Iterating Through Keypoints

You can iterate through keypoints using a loop.

```python
for kp, conf in pose:
    print("Keypoint:", kp)
    print("Confidence:", conf)
```

Example output:

```
Keypoint: [320 240]
Confidence: 0.91
```

Each iteration returns:

| Value | Description |
|------|------|
| `kp` | Keypoint coordinate `[x, y]` |
| `conf` | Confidence score |

---

# Creating Pose from MoveNet Output

The class provides a helper method to convert **MoveNet model outputs**.

```python
pose = Pose.from_movenet(movenet_results)
```

This converts raw MoveNet predictions into a structured `Pose` object.

---

# MoveNet Output Format

MoveNet models output keypoints in the following format:

```
[y, x, confidence]
```

Typical output shapes:

| Shape | Description |
|------|------|
| `(1,1,17,3)` | Nested MoveNet output |
| `(1,17,3)` | Standard MoveNet output |

Where:

| Index | Meaning |
|------|------|
| `0` | y coordinate |
| `1` | x coordinate |
| `2` | confidence score |

---

# How `from_movenet()` Works

The method performs the following steps.

### Step 1 — Remove Nested Containers

Some frameworks return outputs inside lists or tuples.

```
while isinstance(output, (list, tuple)):
    output = output[0]
```

---

### Step 2 — Convert to NumPy Array

The output is converted to a NumPy array.

```
output = np.asarray(output)
```

---

### Step 3 — Extract Keypoints

The method extracts the keypoints depending on the output shape.

```
(1,1,17,3) → output[0][0]
(1,17,3)   → output[0]
```

Resulting shape:

```
(17,3)
```

---

### Step 4 — Separate Coordinates and Confidence

MoveNet outputs values in this order:

```
[y, x, score]
```

The method extracts them separately.

```
y = kps[:,0]
x = kps[:,1]
score = kps[:,2]
```

---

### Step 5 — Convert to (x,y) Format

Coordinates are converted to `[x,y]` format.

```
keypoints = np.stack([x,y], axis=1)
```

Example result:

```
[[x1,y1],
 [x2,y2],
 ...
 [x17,y17]]
```

---

### Step 6 — Create Pose Object

Finally, the method returns a Pose object.

```
return Pose(
    keypoints=keypoints,
    confidence=score
)
```

---

# Example Full Workflow

```python
movenet_output = model(image)

pose = Pose.from_movenet(movenet_output)

for kp, conf in pose:
    print("Keypoint:", kp, "Confidence:", conf)
```

Example output:

```
Keypoint: [320 240] Confidence: 0.92
Keypoint: [330 260] Confidence: 0.88
```

---

# Summary

The `Pose` class provides a simple way to:

- Store pose estimation results
- Access keypoint coordinates
- Access confidence scores
- Iterate through keypoints
- Convert raw MoveNet outputs into structured pose data

This makes it easier to work with **pose estimation models and human keypoint detection results**.