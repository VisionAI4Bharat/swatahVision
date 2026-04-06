import cv2
import numpy as np
from scipy.interpolate import splev, splprep

from swatahvision.annotators.base import BaseAnnotator
from swatahvision.annotators.utils import (
    PENDING_TRACK_ID, 
    ColorLookup, 
    Trace, 
    resolve_color
)
from swatahvision.draw.base import ImageType
from swatahvision.draw.color import Color, ColorPalette
from swatahvision.geometry.core import Position
from swatahvision.core.detection.core import Detections
from swatahvision.utils.conversion import (
    ensure_cv2_image_for_class_method
)


class TraceAnnotator(BaseAnnotator):
    """
    A class for drawing trace paths on an image based on detection coordinates.

    !!! warning

        This annotator uses the `sv.Detections.tracker_id`. Read
        [here](/latest/trackers/) to learn how to plug
        tracking into your inference pipeline.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        position: Position = Position.CENTER,
        trace_length: int = 30,
        thickness: int = 2,
        smooth: bool = False,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color to draw the trace, can be
                a single color or a color palette.
            position (Position): The position of the trace.
                Defaults to `CENTER`.
            trace_length (int): The maximum length of the trace in terms of historical
                points. Defaults to `30`.
            thickness (int): The thickness of the trace lines. Defaults to `2`.
            smooth (bool): Smooth the trace lines.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Color | ColorPalette = color
        self.trace = Trace(max_size=trace_length, anchor=position)
        self.thickness = thickness
        self.smooth = smooth
        self.color_lookup: ColorLookup = color_lookup

    @ensure_cv2_image_for_class_method
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
    ) -> ImageType:
        """
        Draws trace paths on the frame based on the detection coordinates provided.

        Args:
            scene (ImageType): The image on which the traces will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): The detections which include coordinates for
                which the traces will be drawn.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import swatahvision as sv
            from ultralytics import YOLO

            model = YOLO('yolov8x.pt')
            trace_annotator = sv.TraceAnnotator()

            video_info = sv.VideoInfo.from_video_path(video_path='...')
            frames_generator = sv.get_video_frames_generator(source_path='...')
            tracker = sv.ByteTrack()

            with sv.VideoSink(target_path='...', video_info=video_info) as sink:
               for frame in frames_generator:
                   result = model(frame)[0]
                   detections = sv.Detections.from_ultralytics(result)
                   detections = tracker.update_with_detections(detections)
                   annotated_frame = trace_annotator.annotate(
                       scene=frame.copy(),
                       detections=detections)
                   sink.write_frame(frame=annotated_frame)
            ```

        ![trace-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/trace-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        if detections.tracker_id is None:
            raise ValueError(
                "The `tracker_id` field is missing in the provided detections."
                " See more: https://supervision.roboflow.com/latest/how_to/track_objects"
            )
        detections = detections[detections.tracker_id != PENDING_TRACK_ID]

        self.trace.put(detections)
        for detection_idx in range(len(detections)):
            tracker_id = int(detections.tracker_id[detection_idx])
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            xy = self.trace.get(tracker_id=tracker_id)
            spline_points = xy.astype(np.int32)

            if len(xy) > 3 and self.smooth:
                x, y = xy[:, 0], xy[:, 1]
                tck, _u = splprep([x, y], s=20)
                x_new, y_new = splev(np.linspace(0, 1, 100), tck)
                spline_points = np.stack([x_new, y_new], axis=1).astype(np.int32)

            if len(xy) > 1:
                scene = cv2.polylines(
                    scene,
                    [spline_points],
                    False,
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )
        return scene
