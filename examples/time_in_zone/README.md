[SwatahVision - Models Hub](https://huggingface.co/swatah/swatahvision/tree/main)  
[Download Sample Video](https://huggingface.co/datasets/swatah/swatahvision-examples/tree/main/sample-videos) 

# Time in zone

Practical demonstration on leveraging computer vision for analyzing wait times and
monitoring the duration that objects or individuals spend in predefined areas of video
frames. This example project, perfect for retail analytics or traffic management
applications.

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

## scripts

### `draw_zones`

If you want to test zone time in zone analysis on your own video, you can use this
script to design custom zones and save results as a JSON file. The script will open a
window where you can draw polygons on the source image or video file. The polygons will
be saved as a JSON file.

- `--source_path`: Path to the source image or video file for drawing polygons.

- `--zone_configuration_path`: Path where the polygon annotations will be saved as a JSON file.

- `enter` - finish drawing the current polygon.

- `escape` - cancel drawing the current polygon.

- `q` - quit the drawing window.

- `s` - save zone configuration to a JSON file.

```bash
python scripts/draw_zones.py \
    --source_path "data/checkout.mp4" \
    --zone_configuration_path "data/config.json"
```

## run

### `yolov11x-1280 onnx`

Script to run object detection on a video file using the YOLOv11x-1280 onnx model.

- `--zone_configuration_path`: Path to the zone configuration JSON file.
- `--source_video_path`: Path to the source video file.
- `--weights`: Path to the model weights file.
- `--classes`: List of class IDs to track. If empty, all classes are tracked.
- `--confidence_threshold`: Confidence level for detections (`0` to `1`). Default is `0.3`.
- `--iou_threshold`: IOU threshold for non-max suppression. Default is `0.7`.

```bash
python yolov11x-1280_onnx.py \
    --zone_configuration_path "data/config.json" \
    --source_video_path "data/checkout.mp4" \
    --source_weights_path "yolov11x-1280.onnx" \
    --classes 0 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.7
```
