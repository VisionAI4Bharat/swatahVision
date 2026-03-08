# Classification core Module

## Overview

The **Classification module** provides a unified structure for handling classification outputs from deep learning models such as **MobileNet** and **ResNet**.

It converts raw model outputs (logits or probabilities) into a standardized format containing:

- **class_id** → predicted class index
- **confidence** → probability score for the prediction

This allows developers to easily process predictions regardless of which classification model is used.

---

# Class: `Classification`

The `Classification` class stores classification results and provides helper methods to convert outputs from different models into a consistent format.

```python
from dataclasses import dataclass
import numpy as np
```

## Attributes

| Attribute | Type | Description |
|----------|------|-------------|
| `class_id` | `np.ndarray` | Predicted class indices |
| `confidence` | `np.ndarray \| None` | Confidence scores for predictions |

Example structure:

```
class_id = [3]
confidence = [0.92]
```

---

# Number of Predictions

The `__len__()` method returns the number of classifications.

```python
def __len__(self) -> int:
```

### Example

```python
result = Classification(
    class_id=np.array([1, 2, 3]),
    confidence=np.array([0.9, 0.7, 0.6])
)

print(len(result))
```

Output

```
3
```

---

# MobileNet Classification

## Method

```python
Classification.from_mobilenet(mobilenet_results, top_k=None)
```

This method converts **MobileNet model outputs** into a `Classification` object.

### Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `mobilenet_results` | tuple | Model output and metadata |
| `top_k` | int \| None | Number of top predictions to return |

### Behavior

The method automatically:

1. Extracts model outputs
2. Converts tensors to NumPy
3. Applies **softmax** if logits are provided
4. Returns predictions according to `top_k`

---

## Top-K Prediction Options

### 1️⃣ All Classes (Default)

```python
Classification.from_mobilenet(results)
```

Returns probabilities for **all classes**.

Example:

```
class_id = [[0,1,2,3,4]]
confidence = [[0.1,0.3,0.4,0.1,0.1]]
```

---

### 2️⃣ Top-1 Prediction

```python
Classification.from_mobilenet(results, top_k=1)
```

Returns only the **highest probability class**.

Example:

```
class_id = [2]
confidence = [0.91]
```

---

### 3️⃣ Top-K Predictions

```python
Classification.from_mobilenet(results, top_k=3)
```

Returns the **top 3 predictions**.

Example:

```
class_id = [[2,1,4]]
confidence = [[0.91,0.05,0.03]]
```

---

# ResNet Classification

## Method

```python
Classification.from_resnet(resnet_results, top_k=None)
```

This method converts **ResNet model outputs** into a `Classification` object.

The internal processing steps are identical to the MobileNet method:

1. Extract raw outputs
2. Convert tensors to NumPy
3. Normalize outputs into probabilities
4. Select predictions based on `top_k`

---

## Example Usage

### Example 1 — MobileNet

```python
mobilenet_output = model(image)

result = Classification.from_mobilenet(mobilenet_output, top_k=1)

print(result.class_id)
print(result.confidence)
```

Output example

```
class_id = [5]
confidence = [0.94]
```

---

### Example 2 — ResNet

```python
resnet_output = model(image)

result = Classification.from_resnet(resnet_output, top_k=3)

print(result.class_id)
print(result.confidence)
```

Output example

```
class_id = [[3,7,1]]
confidence = [[0.88,0.07,0.02]]
```

---

# Output Format

The module always returns predictions in the following format:

```
Classification(
    class_id = numpy array,
    confidence = numpy array
)
```

Example:

```
Classification(
    class_id=[2],
    confidence=[0.91]
)
```

---

# Supported Model Types

Currently supported models:

- MobileNet
- ResNet

Additional classification models can be added by implementing similar conversion methods.

---

# Error Handling

The module raises errors in the following cases:

### Invalid Output Shape

```
ValueError: Invalid output shape
```

This occurs if the model output is not compatible with the expected format `(B, C)`.

---

### Invalid `top_k`

```
ValueError: top_k cannot be greater than number of classes
```

This prevents requesting more predictions than available classes.

---

# Summary

The `Classification` module provides:

- Standardized output format
- Support for MobileNet and ResNet
- Automatic tensor to NumPy conversion
- Optional Top-K prediction support
- Easy integration into inference pipelines

This abstraction simplifies the handling of classification outputs and ensures consistent processing across different models.

---

# Authors

**Atharva Kotkar**  
**Aarav Agrawal**  
MIT Internship – Swatah AI