<p align="center">
  <img src="assets/cover.png" alt="SwatahVision Cover" width="100%">
</p>

swatahVision
============

**swatahVision** is a lightweight Python computer vision library designed to simplify **model loading**, **inference**, and **visualization** for image classification and object detection tasks.

The library provides a clean and minimal API that allows developers to run vision models with very little code.

Installation
------------

    pip install swatahvision

Install from source:

    git clone https://github.com/VisionAI4Bharat/swatahVision.git
    cd swatahVision
    pip install -e .

Documentation
-------------

Full package documentation is available at:

[https://visionai4bharat.github.io/swatahVision](https://visionai4bharat.github.io/swatahVision)

Image Classification Example
----------------------------

    import swatahVision as sv
    
    model = sv.Model(
        model="mobilenetv2-10",
        engine=sv.Engine.ONNX,
        hardware=sv.Hardware.CPU,
    )
    
    image = sv.Image.load_from_file("assets/car.jpg")
    
    outs = model(image)
    
    classification = sv.Classification.from_mobilenet(
        outs,
        top_k=5,
    )
    
    print(classification)
    

Object Detection Example
------------------------

    import swatahVision as sv
    
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.YELLOW,
        text_color=sv.Color.BLACK,
        text_position=sv.Position.TOP_LEFT,
        text_scale=0.7,
        text_padding=8,
        smart_position=False,
    )
    
    box_annotator = sv.BoxAnnotator(sv.Color.YELLOW)
    
    model = sv.Model(
        model="ssdlite-mobilenetv3-13",
        engine=sv.Engine.ONNX,
        hardware=sv.Hardware.CPU,
    )
    
    image = sv.Image.load_from_file("assets/car.jpg")
    
    outs = model(image)
    
    detections = sv.Detections.from_ssd(
        outs,
        conf_threshold=0.5,
    )
    
    image = label_annotator.annotate(scene=image, detections=detections)
    image = box_annotator.annotate(scene=image, detections=detections)
    
    sv.Image.show(image=image)
    

Features
--------

*   Unified model loading interface
*   ONNX Runtime inference
*   CPU-friendly execution
*   Image classification support
*   Object detection support
*   Built-in annotation utilities
*   Minimal and clean API

Core Components
---------------

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

Design Philosophy
-----------------

*   Simplicity
*   Lightweight deployment
*   Fast prototyping
*   Developer-friendly workflows

License
-------

All work under VisionAI4Bhārat is released under permissive open-source licenses (MIT, Apache 2.0, or similar). Each repository or project may use a different license - please check the repository license file.