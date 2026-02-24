# Pose Estimation
This demo performs human pose estimation using the MoveNet model. It detects 17 body keypoints for each person in an image or video and provides their coordinates along with confidence scores. The detected keypoints can be used for visualization, tracking, or further analysis.

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

- movenet onnx

    - `--source_weights_path` (optional): The path to the MoveNet model file (ONNX / TFLite / TensorFlow SavedModel).
        Defaults to `"movenet.onnx"` if not specified.

    - `--source_video_path`: The path to the input image or video file for pose estimation.
        This can be:

        - An image (.jpg, .png)

        - A video (.mp4, .avi)

        - Or camera index (0 for webcam)

    - `--target_video_path` (optional): The path to save the output image or video with pose keypoints drawn.
        If not provided, the output will be displayed in real-time.

    - `--confidence_threshold` (optional): Sets the minimum keypoint confidence score.
        Keypoints with confidence below this value will be ignored.
        Default is `0.3`.

    - `--normalize` (optional): Whether to normalize input image before inference.
        Default: `True`

## run example

- movenet onnx

    ```bash
    python movenet_pose_onnx.py \
        --source_weights_path data/movenet.onnx \
        --source_video_path data/input.mp4 \
        --output_path outputs/output.mp4 \
        --confidence_threshold 0.3
    ```
