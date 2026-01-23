import argparse

import cv2
import numpy as np

from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import swatahvision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


def main(
    source_video_path: str,
    zone_configuration_path: str,
    source_weights_path: str,
    confidence: float,
    iou: float,
    classes: list[int],
) -> None:
    model = sv.Model(model=source_weights_path, engine=sv.Engine.ONNX, hardware=sv.Hardware.GPU)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)
    
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    for frame in frames_generator:
        results = model(frame)
        detections = sv.Detections.from_yolo(results, conf_threshold=confidence, nms_threshold=iou, class_agnostic=True)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()

        for idx, zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

        video_writer.write(annotated_frame)
        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using video file."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--source_weights_path", type=str, default="yolov8n-13", help="swatah vision model."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        source_weights_path=args.source_weights_path,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )