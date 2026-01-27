import argparse

from tqdm import tqdm

import swatahvision as sv

def process_video(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    model = sv.Model(model=source_weights_path, engine=sv.Engine.ONNX, hardware=sv.Hardware.GPU)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            results = model(frame)
            detections = sv.Detections.from_yolo(results, conf_threshold=confidence_threshold, nms_threshold=iou_threshold, class_agnostic=True)
            filter_by_class = detections.class_id == 0
            detections = detections[filter_by_class]
            detections = tracker.update_with_detections(detections)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )
            
            labels = [
                f"ID {tracker_id}" for tracker_id in detections.tracker_id
            ]

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            sink.write_frame(frame=annotated_labeled_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Processing with YOLO and ByteTrack"
    )
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()

    process_video(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )