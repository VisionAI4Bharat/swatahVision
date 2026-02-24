import argparse
import cv2
import numpy as np
import swatahvision as sv


# ---------------- MoveNet Output Parser ----------------
def parse_movenet(output: any) -> np.ndarray:
    """
    Convert MoveNet output to (17,3) → x, y, confidence
    Supports shapes:
        (1,1,17,3)
        (1,17,3)
    """
    while isinstance(output, (list, tuple)):
        output = output[0]

    output = np.asarray(output)

    if output.ndim == 4:
        keypoints = output[0][0]
    elif output.ndim == 3:
        keypoints = output[0]
    else:
        raise ValueError(f"Invalid MoveNet output shape: {output.shape}")

    # MoveNet format: y, x, score
    y = keypoints[:, 0]
    x = keypoints[:, 1]
    score = keypoints[:, 2]

    return np.stack([x, y, score], axis=1)


# ---------------- Preprocess ----------------
def preprocess(frame: np.ndarray, input_size: int = 192):
    """
    Resize and normalize frame for MoveNet
    """
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, (w, h)


# ---------------- Draw Keypoints ----------------
def draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    frame_size: tuple[int, int],
    conf_threshold: float,
) -> np.ndarray:
    """
    Draw visible keypoints on frame
    """
    w, h = frame_size

    for x, y, conf in keypoints:
        if conf >= conf_threshold:
            px = int(x * w)
            py = int(y * h)
            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    return frame


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pose Estimation using MoveNet ONNX with SwatahVision"
    )

    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to movenet.onnx",
    )

    parser.add_argument(
        "--source_video_path",
        required=True,
        type=str,
        help="Path to input video",
    )

    parser.add_argument(
        "--target_video_path",
        default=None,
        type=str,
        help="Path to save output video (optional)",
    )

    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        type=float,
        help="Minimum keypoint confidence",
    )

    args = parser.parse_args()

    # Load ONNX model
    model = sv.Model(
        model=args.model_path,
        engine=sv.Engine.ONNX,
        hardware=sv.Hardware.CPU,
    )

    # Video info
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    frames_generator = sv.get_video_frames_generator(args.source_video_path)

    # -------- Save to file --------
    if args.target_video_path:
        with sv.VideoSink(args.target_video_path, video_info) as sink:
            for frame in frames_generator:
                input_tensor, frame_size = preprocess(frame)
                results = model(input_tensor)

                keypoints = parse_movenet(results)
                annotated_frame = draw_keypoints(
                    frame, keypoints, frame_size, args.confidence_threshold
                )

                sink.write_frame(annotated_frame)

    # -------- Display --------
    else:
        for frame in frames_generator:
            input_tensor, frame_size = preprocess(frame)
            results = model(input_tensor)

            keypoints = parse_movenet(results)
            annotated_frame = draw_keypoints(
                frame, keypoints, frame_size, args.confidence_threshold
            )

            cv2.imshow("Pose Estimation", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()