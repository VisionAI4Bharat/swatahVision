[SwatahVision - Models Hub](https://huggingface.co/swatah/swatahvision/tree/main)  
[Download Sample Video](https://huggingface.co/datasets/swatah/swatahvision-examples/tree/main/sample-videos) 

# Count people in zone

This demo is a video analysis tool that counts and highlights objects in specific zones
of a video. Each zone and the objects within it are marked in different colors, making
it easy to see and count the objects in each area. The tool can save this enhanced
video or display it live on the screen.

## install

- clone repository and navigate to example directory

    ```bash
    git clone https://github.com/VisionAI4Bharat/swatahvision.git
    cd swatahvision/examples/count_people_in_zone
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

    - `--source_weights_path` (optional): The path to the YOLO model's weights file.
        Defaults to `"yolov11x-1280.onnx"` if not specified.

    - `--zone_configuration_path`: Specifies the path to the JSON file containing zone
        configurations. This file defines the polygonal areas in the video where objects will
        be counted.

    - `--source_video_path`: The path to the source video file that will be analyzed.

    - `--target_video_path` (optional): The path to save the output video with annotations.
        If not provided, the processed video will be displayed in real-time.

    - `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO model
        to filter detections. Default is `0.3`.

    - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
        for the model. Default is `0.7`.

## zone configuration

- `horizontal-zone-config.json`: Defines zones divided horizontally across the frame.
- `multi-zone-config.json`: Configures multiple zones with custom shapes and positions.
- `quarters-zone-config.json`: Splits the frame into four equal quarters.
- `vertical-zone-config.json`: Divides the frame into vertical zones of equal width.

## run example

- yolov11x-1280 onnx

    ```bash
    python yolov11x-1280_onnx.py \
        --source_weights_path data/yolov11x-1280.onnx
        --zone_configuration_path data/multi-zone-config.json \
        --source_video_path data/market-square.mp4 \
        --confidence_threshold 0.3 \
        --iou_threshold 0.5
    ```
