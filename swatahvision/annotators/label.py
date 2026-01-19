import numpy as np

from swatahvision.annotators.base import BaseAnnotator
from swatahvision.annotators.utils import (
    PENDING_TRACK_ID,
    ColorLookup,
    Trace,
    get_labels_text,
    resolve_color,
    resolve_text_background_xyxy,
    snap_boxes,
    validate_labels,
    wrap_text,
)
from swatahvision.draw.color import Color, ColorPalette
from swatahvision.draw.base import ImageType

from swatahvision.core.detection.core import Detections
from swatahvision.core.detection.utils.boxes import clip_boxes, spread_out_boxes
from swatahvision.geometry.core import Position

import cv2

CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX



class _BaseLabelAnnotator(BaseAnnotator):
    """
    Base class for annotators that add labels to detections.

    Attributes:
        color (Union[Color, ColorPalette]): The color to use for the label background.
        color_lookup (ColorLookup): The method used to determine the color of the label.
        text_color (Union[Color, ColorPalette]): The color to use for the label text.
        text_padding (int): The padding around the label text, in pixels.
        text_anchor (Position): The position of the text relative to the detection
            bounding box.
        text_offset (Tuple[int, int]): A tuple of 2D coordinates `(x, y)` to
            offset the text position from the anchor point, in pixels.
        border_radius (int): The radius of the label background corners, in pixels.
        smart_position (bool): Whether to intelligently adjust the label position to
            avoid overlapping with other elements.
        max_line_length (Optional[int]): Maximum number of characters per line before
            wrapping the text. None means no wrapping.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        text_color: Color | ColorPalette = Color.WHITE,
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        text_offset: tuple[int, int] = (0, 0),
        border_radius: int = 0,
        smart_position: bool = False,
        max_line_length: int | None = None,
    ):
        """
        Initializes the _BaseLabelAnnotator.

        Args:
            color (Union[Color, ColorPalette], optional): The color to use for the label
                background.
            color_lookup (ColorLookup, optional): The method used to determine the color
                of the label
            text_color (Union[Color, ColorPalette], optional): The color to use for the
                label text.
            text_padding (int, optional): The padding around the label text, in pixels.
            text_position (Position, optional): The position of the text relative to the
                detection bounding box.
            text_offset (Tuple[int, int], optional): A tuple of 2D coordinates
                `(x, y)` to offset the text position from the anchor point, in pixels.
            border_radius (int, optional): The radius of the label background corners,
                in pixels.
            smart_position (bool, optional): Whether to intelligently adjust the label
                position to avoid overlapping with other elements.
            max_line_length (Optional[int], optional): Maximum number of characters per
                line before wrapping the text. None means no wrapping.
        """
        self.color: Color | ColorPalette = color
        self.color_lookup: ColorLookup = color_lookup
        self.text_color: Color | ColorPalette = text_color
        self.text_padding: int = text_padding
        self.text_anchor: Position = text_position
        self.text_offset: tuple[int, int] = text_offset
        self.border_radius: int = border_radius
        self.smart_position = smart_position
        self.max_line_length: int | None = max_line_length

    def _adjust_labels_in_frame(
        self,
        resolution_wh: tuple[int, int],
        labels: list[str],
        label_properties: np.ndarray,
    ) -> np.ndarray:
        """
        Adjusts the position of labels to ensure they stay within the frame boundaries.

        Args:
            frame_width (int): The width of the frame.
            resolution_wh (int, int): The width and height of the frame.
            labels (List[str]): The list of text labels.
            label_properties (np.ndarray): An array of label properties, where each row
                            contains [x1, y1, x2, y2, text_height, ...].

        Returns:
            np.ndarray: The adjusted label properties.
        """
        adjusted_properties = label_properties.copy()

        # First, make sure the boxes don't go outside the frame
        adjusted_properties[:, :4] = snap_boxes(
            adjusted_properties[:, :4],
            resolution_wh,
        )

        # Apply the spread out algorithm to avoid box overlaps
        if len(labels) > 1:
            # Extract the box coordinates
            boxes = adjusted_properties[:, :4]
            # Use the spread_out_boxes function to adjust overlapping boxes
            spread_boxes = spread_out_boxes(boxes)
            # Update the properties with the spread out boxes
            adjusted_properties[:, :4] = spread_boxes

            # Additional check to ensure boxes are still within frame after spreading
            adjusted_properties[:, :4] = snap_boxes(
                adjusted_properties[:, :4], resolution_wh
            )

        return adjusted_properties



class LabelAnnotator(_BaseLabelAnnotator):
    """
    A class for annotating labels on an image using provided detections.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        text_color: Color | ColorPalette = Color.WHITE,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        text_offset: tuple[int, int] = (0, 0),
        border_radius: int = 0,
        smart_position: bool = False,
        max_line_length: int | None = None,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating the text background.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            text_color (Union[Color, ColorPalette]): The color or color palette to use
                for the text.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_padding (int): Padding around the text within its background box.
            text_position (Position): Position of the text relative to the detection.
                Possible values are defined in the `Position` enum.
            text_offset (Tuple[int, int]): A tuple of 2D coordinates `(x, y)` to
                offset the text position from the anchor point, in pixels.
            border_radius (int): The radius to apply round edges. If the selected
                value is higher than the lower dimension, width or height, is clipped.
            smart_position (bool): Spread out the labels to avoid overlapping.
            max_line_length (Optional[int]): Maximum number of characters per line
                before wrapping the text. None means no wrapping.
        """
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        super().__init__(
            color=color,
            color_lookup=color_lookup,
            text_color=text_color,
            text_padding=text_padding,
            text_position=text_position,
            text_offset=text_offset,
            border_radius=border_radius,
            smart_position=smart_position,
            max_line_length=max_line_length,
        )

    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        labels: list[str] | None = None,
        custom_color_lookup: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with labels based on the provided detections.

        Args:
            scene (ImageType): The image where labels will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            labels (Optional[List[str]]): Custom labels for each detection.
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

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(detections['class_name'], detections.confidence)
            ]

            label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
            annotated_frame = label_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                labels=labels
            )
            ```

        ![label-annotator-example](https://media.roboflow.com/
        swatahvision-annotator-examples/label-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        validate_labels(labels, detections)

        labels = get_labels_text(detections, labels)
        label_properties = self._get_label_properties(detections, labels)

        if self.smart_position:
            xyxy = label_properties[:, :4]
            xyxy = spread_out_boxes(xyxy)
            label_properties[:, :4] = xyxy

            label_properties = self._adjust_labels_in_frame(
                (scene.shape[1], scene.shape[0]),
                labels,
                label_properties,
            )

        self._draw_labels(
            scene=scene,
            labels=labels,
            label_properties=label_properties,
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )

        return scene

    def _get_label_properties(
        self,
        detections: Detections,
        labels: list[str],
    ) -> np.ndarray:
        label_properties = []
        anchors_coordinates = detections.get_anchors_coordinates(
            anchor=self.text_anchor
        ).astype(int)

        for label, center_coordinates in zip(labels, anchors_coordinates):
            center_coordinates = (
                center_coordinates[0] + self.text_offset[0],
                center_coordinates[1] + self.text_offset[1],
            )

            wrapped_lines = wrap_text(label, self.max_line_length)
            line_heights = []
            line_widths = []

            for line in wrapped_lines:
                (text_w, text_h) = cv2.getTextSize(
                    text=line,
                    fontFace=CV2_FONT,
                    fontScale=self.text_scale,
                    thickness=self.text_thickness,
                )[0]
                line_heights.append(text_h)
                line_widths.append(text_w)

            # Get the maximum width and total height
            max_width = max(line_widths) if line_widths else 0
            total_height = (
                sum(line_heights) + (len(line_heights) - 1) * self.text_padding
            )

            # Add padding around all sides
            width_padded = max_width + 2 * self.text_padding
            height_padded = total_height + 2 * self.text_padding

            text_background_xyxy = resolve_text_background_xyxy(
                center_coordinates=center_coordinates,
                text_wh=(width_padded, height_padded),
                position=self.text_anchor,
            )

            label_properties.append(
                [
                    *text_background_xyxy,
                    total_height,
                ]
            )
        return np.array(label_properties).reshape(-1, 5)

    def _draw_labels(
        self,
        scene: np.ndarray,
        labels: list[str],
        label_properties: np.ndarray,
        detections: Detections,
        custom_color_lookup: np.ndarray | None,
    ) -> None:
        assert len(labels) == len(label_properties) == len(detections), (
            f"Number of label properties ({len(label_properties)}), "
            f"labels ({len(labels)}) and detections ({len(detections)}) "
            "do not match."
        )

        color_lookup = (
            custom_color_lookup
            if custom_color_lookup is not None
            else self.color_lookup
        )

        for idx, label_property in enumerate(label_properties):
            background_color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=idx,
                color_lookup=color_lookup,
            )
            text_color = resolve_color(
                color=self.text_color,
                detections=detections,
                detection_idx=idx,
                color_lookup=color_lookup,
            )

            box_xyxy = label_property[:4].astype(int)

            self.draw_rounded_rectangle(
                scene=scene,
                xyxy=box_xyxy,
                color=background_color.as_bgr(),
                border_radius=self.border_radius,
            )

            # Handle multiline text
            wrapped_lines = wrap_text(labels[idx], self.max_line_length)
            current_y = box_xyxy[1] + self.text_padding  # Start y position

            for line in wrapped_lines:
                if not line:
                    # Use a character with ascenders and descenders as height reference
                    (_, text_h) = cv2.getTextSize(
                        text="Tg",
                        fontFace=CV2_FONT,
                        fontScale=self.text_scale,
                        thickness=self.text_thickness,
                    )[0]
                    current_y += text_h + self.text_padding
                    continue

                (_, text_h) = cv2.getTextSize(
                    text=line,
                    fontFace=CV2_FONT,
                    fontScale=self.text_scale,
                    thickness=self.text_thickness,
                )[0]

                text_x = box_xyxy[0] + self.text_padding
                text_y = current_y + text_h  # Add height to get to text baseline

                cv2.putText(
                    img=scene,
                    text=line,
                    org=(text_x, text_y),
                    fontFace=CV2_FONT,
                    fontScale=self.text_scale,
                    color=text_color.as_bgr(),
                    thickness=self.text_thickness,
                    lineType=cv2.LINE_AA,
                )

                current_y += text_h + self.text_padding  # Move to next line position

    @staticmethod
    def draw_rounded_rectangle(
        scene: np.ndarray,
        xyxy: tuple[int, int, int, int],
        color: tuple[int, int, int],
        border_radius: int,
    ) -> np.ndarray:
        x1, y1, x2, y2 = xyxy
        width = x2 - x1
        height = y2 - y1

        border_radius = min(border_radius, min(width, height) // 2)

        rectangle_coordinates = [
            ((x1 + border_radius, y1), (x2 - border_radius, y2)),
            ((x1, y1 + border_radius), (x2, y2 - border_radius)),
        ]
        circle_centers = [
            (x1 + border_radius, y1 + border_radius),
            (x2 - border_radius, y1 + border_radius),
            (x1 + border_radius, y2 - border_radius),
            (x2 - border_radius, y2 - border_radius),
        ]

        for coordinates in rectangle_coordinates:
            cv2.rectangle(
                img=scene,
                pt1=coordinates[0],
                pt2=coordinates[1],
                color=color,
                thickness=-1,
            )
        for center in circle_centers:
            cv2.circle(
                img=scene,
                center=center,
                radius=border_radius,
                color=color,
                thickness=-1,
            )
        return scene