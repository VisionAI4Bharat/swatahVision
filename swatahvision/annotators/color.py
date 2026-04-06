import cv2
import numpy as np
from swatahvision.annotators.base import BaseAnnotator
from swatahvision.draw.color import ColorPalette, Color
from swatahvision.annotators.utils import ColorLookup, resolve_color
from swatahvision.core.detection.core import Detections
from swatahvision.utils.conversion import (
    ensure_cv2_image_for_class_method,
)
from swatahvision.draw.base import ImageType

class ColorAnnotator(BaseAnnotator):
    """
    A class for drawing box masks on an image using provided detections.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        opacity: float = 0.5,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            opacity (float): Opacity of the overlay mask. Must be between `0` and `1`.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Color | ColorPalette = color
        self.color_lookup: ColorLookup = color_lookup
        self.opacity = opacity

    @ensure_cv2_image_for_class_method
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
    ) -> ImageType:
        """
        Annotates the given scene with box masks based on the provided detections.

        Args:
            scene (ImageType): The image where bounding boxes will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            color_annotator = sv.ColorAnnotator()
            annotated_frame = color_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![box-mask-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/box-mask-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        scene_with_boxes = scene.copy()
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            cv2.rectangle(
                img=scene_with_boxes,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=-1,
            )

        cv2.addWeighted(
            scene_with_boxes, self.opacity, scene, 1 - self.opacity, gamma=0, dst=scene
        )
        return scene