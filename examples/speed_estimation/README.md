[SwatahVision - Models Hub](https://huggingface.co/swatah/swatahvision/tree/main)  
[Download Sample Video](https://huggingface.co/datasets/swatah/swatahvision-examples/tree/main/sample-videos) 


# Speed Estimation

This example performs speed estimation analysis using various object-detection models
and ByteTrack - a simple yet effective online multi-object tracking method. It uses the
swatahvision package for multiple tasks such as tracking, annotations, etc.

## install

- clone repository and navigate to example directory

    ```bash
    git clone https://github.com/VisionAI4Bharat/swatahvision.git
    cd swatahvision/examples/speed-estimation    
    ```

- setup python environment and activate it [optional]

    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    ```

- install required dependencies

    ```bash
    pip install -r requirements.txt
    ```

## script arguments

- `--source_weights_path`: Required. Specifies the path to the YOLO model's weights (yolov11x-1280)
    file, which is essential for the object detection process. This file contains the
    data that the model uses to identify objects in the video.

- `--source_video_path`: Required. The path to the source video file that will be
    analyzed. This is the input video on which traffic flow analysis will be performed.

- `--target_video_path`: The path to save the output video with
    annotations. If not specified, the processed video will be displayed in real-time
    without being saved.

- `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO
    model to filter detections. Default is `0.3`. This determines how confident the
    model should be to recognize an object in the video.

- `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
    for the model. Default is 0.7. This value is used to manage object detection
    accuracy, particularly in distinguishing between different objects.

## run

- yolov11x-1280 onnx

    ```bash
    python yolov11x-1280_onnx.py \
        --source_weights_path models/yolov11x-1280.onnx \
        --source_video_path data/vehicles.mp4 \
        --target_video_path data/vehicles-result.mp4 \
        --confidence_threshold 0.3 \
        --iou_threshold 0.5
    ```

- yolov11x-1280 openvino

    ```bash
    python yolov11x-1280_openvino.py \
        --source_weights_path models/yolov11x-1280.xml \
        --source_video_path data/vehicles.mp4 \
        --target_video_path data/vehicles-result.mp4 \
        --confidence_threshold 0.3 \
        --iou_threshold 0.5
    ```
