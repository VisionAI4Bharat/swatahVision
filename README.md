<p align="center">
  <img src="assets/cover.png" alt="SwatahVision Cover" width="100%">
</p>


### An open-source Vision AI stack for real-world applications

**swatahVision** is an **open-source Vision AI stack** that brings together **models, runtimes, post-processing, tracking, and visualization** into a clean, reusable Python package.

It’s built to make Vision AI **practical**: load a model, run inference, get structured outputs, visualize results, and ship pipelines faster — without reinventing glue-code every time.

> Part of the **VisionAI4Bhārat** initiative.

---

## What is swatahVision?

swatahVision provides a unified interface for:

- **Inference** across multiple runtimes (e.g., **ONNX Runtime**, **OpenVINO**)
- **Vision tasks**
  - Object **Detection**
  - Image **Classification**
  - (Extensible for OCR / segmentation / multimodal pipelines)
- **Post-processing adapters**
  - YOLO-style decoders + NMS
  - SSD-style decoders
  - RetinaNet-style decoders
- **Tracking**
  - Integrated **ByteTrack** support to assign stable `tracker_id` across frames
- **Visualization**
  - Built-in drawing utilities for boxes, labels, and overlays
- **Video processing utilities**
  - Frame generators and pipeline helpers

---

## Features

- ✅ **Unified model wrapper**: same API for different backends/runtimes  
- ✅ **Structured outputs**:
  - `Detections`: boxes, confidence, class_id, tracker_id (and more)
  - `Classification`: class_id, confidence, top-k
- ✅ **Production-friendly utilities**: FPS monitor, video readers/writers, batching-ready patterns
- ✅ **Lightweight & composable**: designed as a stack, not a monolith
- ✅ **Real world examples and analytics**
- ✅ **Built-in annotation utilities
- ✅ **Minimal and clean API
---
## Core Components


### Model

    sv.Model(model, engine, hardware)

Handles model loading and inference execution.

### Image API

    image = sv.Image.load_from_file(path)
    sv.Image.show(image)

### Post Processing

    sv.Classification.from_mobilenet(...)
    sv.Detections.from_ssd(...)

### Annotation

*   Bounding boxes
*   Labels
*   Custom colors
*   Text positioning

## Design Philosophy

*   Simplicity
*   Lightweight deployment
*   Fast prototyping
*   Developer-friendly workflows

## Install

### From source
```bash
git clone https://github.com/VisionAI4Bharat/swatahvision.git
cd swatahvision
pip install -e .
```
## Quickstart
### Load a model and run inference
```
import swatahvision as sv

model = sv.Model(
    model="path/to/model.onnx",
    engine=sv.Engine.ONNX,
    hardware=sv.Hardware.CPU
)

outputs = model(image, input_size=(640, 640))
```
### Detection
#### Convert raw outputs to Detections
```
import swatahvision as sv

detections = sv.Detections.from_yolo(
    outputs,
    conf_threshold=0.3,
    nms_threshold=0.5,
    class_agnostic=False
)

print(len(detections))
print(detections.xyxy[:3])
```
#### Filter / slice detections
```
high_conf = detections[detections.confidence > 0.6]
persons = detections[detections.class_id == 0]
```
#### Draw boxes
```
import cv2
import swatahvision as sv

frame = cv2.imread("image.jpg")

annotated = sv.UI.draw_bboxes(
    image=frame.copy(),
    detections=detections,
    conf=0.3
)

cv2.imwrite("out.jpg", annotated)
```
## License

This work is licensed under LGPL 3.0
