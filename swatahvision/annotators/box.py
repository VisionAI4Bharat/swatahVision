import numpy as np

from swatahvision.annotators.base import BaseAnnotator
from swatahvision.annotators.utils import ColorLookup, resolve_color
from swatahvision.draw.color import Color, ColorPalette
from swatahvision.draw.base import ImageType

from swatahvision.core.detection.core import Detections

import cv2


class BoxAnnotator(BaseAnnotator):
    """
    A class for drawing bounding boxes on an image using provided detections.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the bounding box lines.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Color | ColorPalette = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: np.ndarray | None = None,
    ) -> ImageType:
        """
        Annotates the given scene with bounding boxes based on the provided detections.

        Args:
            scene (ImageType): The image where bounding boxes will be drawn. `ImageType`
                is a flexible type, accepting either `numpy.ndarray` or
                `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import swatahvision as sv

            image = ...
            detections = sv.Detections(...)

            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![bounding-box-annotator-example](https://media.roboflow.com/
        swatahvision-annotator-examples/bounding-box-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
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
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
        return scene