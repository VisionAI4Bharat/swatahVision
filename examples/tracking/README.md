[SwatahVision - Models Hub](https://huggingface.co/swatah/swatahvision/tree/main)  
[Download Sample Video](https://huggingface.co/datasets/swatah/swatahvision-examples/tree/main/sample-videos) 

# Tracking

This script provides functionality for processing videos using YOLOv11x-1280 for object detection and Swatahvision for tracking and annotation.

## install

- clone repository and navigate to example directory

    ```bash
    git clone https://github.com/VisionAI4Bharat/swatahvision.git
    cd swatahvision/examples/speed-estimation    
    ```

- setup python environment and activate it [optional]

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

- install required dependencies

    ```bash
    pip install -r requirements.txt
    ```

## script arguments

- yolov11x-1280 onnx

    - `--source_weights_path`: Required. Specifies the path to the YOLO model's weights (yolov11x-1280.onnx)
        file, which is essential for the object detection process. This file contains the data
        that the model uses to identify objects in the video.

    - `--source_video_path`: Required. The path to the source video file to be processed.
        This is the video on which object detection and annotation will be performed.

    - `--target_video_path`: Required. The path where the processed video, with annotations
        added, will be saved. This is your output video file.

    - `--confidence_threshold` (optional): Sets the confidence level at which the model
        identifies objects in the video. Default is `0.3`. A higher threshold makes the model
        more selective, while a lower threshold makes it more inclusive in identifying objects.

    - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
        for the model, defaulting to `0.7`. This parameter helps in differentiating between
        distinct objects, especially in crowded scenes.

## run

- yolov11x-1280 onnx

    ```bash
    python yolov11x-1280_onnx.py \
        --source_weights_path yolov11x-1280.onnx \
        --source_video_path input.mp4 \
        --target_video_path tracking_result.mp4
    ```