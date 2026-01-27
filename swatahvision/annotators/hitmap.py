import cv2
import numpy as np
import numpy.typing as npt
from swatahvision.annotators.base import BaseAnnotator
from swatahvision.geometry.core import Position
from swatahvision.core.detection.core import Detections
from swatahvision.utils.conversion import (
    ensure_cv2_image_for_class_method,
)
from swatahvision.draw.base import ImageType


class HeatMapAnnotator(BaseAnnotator):
    """
    A class for drawing heatmaps on an image based on provided detections.
    Heat accumulates over time and is drawn as a semi-transparent overlay
    of blurred circles.
    """

    def __init__(
        self,
        position: Position = Position.BOTTOM_CENTER,
        opacity: float = 0.2,
        radius: int = 40,
        kernel_size: int = 25,
        top_hue: int = 0,
        low_hue: int = 125,
    ):
        """
        Args:
            position (Position): The position of the heatmap. Defaults to
                `BOTTOM_CENTER`.
            opacity (float): Opacity of the overlay mask, between 0 and 1.
            radius (int): Radius of the heat circle.
            kernel_size (int): Kernel size for blurring the heatmap.
            top_hue (int): Hue at the top of the heatmap. Defaults to 0 (red).
            low_hue (int): Hue at the bottom of the heatmap. Defaults to 125 (blue).
        """
        self.position = position
        self.opacity = opacity
        self.radius = radius
        self.kernel_size = kernel_size
        self.top_hue = top_hue
        self.low_hue = low_hue
        self.heat_mask: npt.NDArray[np.float32] | None = None

    @ensure_cv2_image_for_class_method
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:
        """
        Annotates the scene with a heatmap based on the provided detections.

        Args:
            scene (ImageType): The image where the heatmap will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv
            from ultralytics import YOLO

            model = YOLO('yolov8x.pt')

            heat_map_annotator = sv.HeatMapAnnotator()

            video_info = sv.VideoInfo.from_video_path(video_path='...')
            frames_generator = sv.get_video_frames_generator(source_path='...')

            with sv.VideoSink(target_path='...', video_info=video_info) as sink:
               for frame in frames_generator:
                   result = model(frame)[0]
                   detections = sv.Detections.from_ultralytics(result)
                   annotated_frame = heat_map_annotator.annotate(
                       scene=frame.copy(),
                       detections=detections)
                   sink.write_frame(frame=annotated_frame)
            ```

        ![heatmap-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/heat-map-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        if self.heat_mask is None:
            self.heat_mask = np.zeros(scene.shape[:2], dtype=np.float32)

        mask = np.zeros(scene.shape[:2])
        for xy in detections.get_anchors_coordinates(self.position):
            x, y = int(xy[0]), int(xy[1])
            cv2.circle(
                img=mask,
                center=(x, y),
                radius=self.radius,
                color=(1,),
                thickness=-1,  # fill
            )
        self.heat_mask = mask + self.heat_mask
        temp = self.heat_mask.copy()
        temp = self.low_hue - temp / temp.max() * (self.low_hue - self.top_hue)
        temp = temp.astype(np.uint8)
        if self.kernel_size is not None:
            temp = cv2.blur(temp, (self.kernel_size, self.kernel_size))
        hsv = np.full(scene.shape, 255, dtype=np.uint8)
        hsv[..., 0] = temp
        temp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mask = cv2.cvtColor(self.heat_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR) > 0
        scene[mask] = cv2.addWeighted(temp, self.opacity, scene, 1 - self.opacity, 0)[
            mask
        ]
        return scene