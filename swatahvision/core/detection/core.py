from dataclasses import dataclass, field
from typing import Any, Iterator, Self
from functools import reduce
import numpy as np
import cv2

from swatahvision.core.detection.utils.internal import (
    is_data_equal, 
    is_metadata_equal, 
    get_data_item, 
    merge_data, 
    merge_metadata,
    )
from swatahvision.core.detection.utils.iou_and_nms import (
    OverlapMetric,
    box_iou_batch,
    box_non_max_merge,
    box_non_max_suppression,
    mask_iou_batch,
    mask_non_max_merge,
    mask_non_max_suppression,
)

from swatahvision.core.validators import validate_detections_fields
from swatahvision.core.utils.internal import deprecated, get_instance_variables
from swatahvision.geometry.core import Position

from swatahvision.core.detection.utils.masks import calculate_masks_centroids

def scale_boxes(boxes, meta):
    scale, pad_x, pad_y = meta

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= scale

    return boxes


@dataclass
class Detections:
    xyxy: np.ndarray
    mask: np.ndarray | None = None
    confidence: np.ndarray | None = None
    class_id: np.ndarray | None = None
    tracker_id: np.ndarray | None = None
    data: dict[str, np.ndarray | list] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        validate_detections_fields(
            xyxy=self.xyxy,
            mask=self.mask,
            confidence=self.confidence,
            class_id=self.class_id,
            tracker_id=self.tracker_id,
            data=self.data,
    )
    
    @classmethod
    def from_yolo(cls, yolo_results, conf_threshold: int=None, nms_threshold: float=None, class_agnostic: bool=False) -> Self:
        raw_outputs, meta = yolo_results
        
        output = raw_outputs[0]
        pred = output[0].transpose(1, 0)
        
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)

        if conf_threshold is not None:
            mask = confidences > conf_threshold
            boxes = boxes[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]

        cx, cy, w, h = boxes.T
        boxes = np.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        axis=1
        )
        
        boxes = scale_boxes(boxes=boxes, meta=meta)
        
        detections = cls(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )
        
        if nms_threshold is not None:
            detections = detections.with_nms(
                threshold=nms_threshold,
                class_agnostic=class_agnostic,
                overlap_metric=OverlapMetric.IOU,
            )
        
        return detections
    
    @classmethod
    def from_ssd(cls, ssd_results, conf_threshold: int=0.5) -> Self:
        boxes, confidences, class_ids = ssd_results[0]
        
        mask = confidences > conf_threshold
        
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        meta = ssd_results[1]
        boxes = scale_boxes(boxes=boxes, meta=meta)
        
        return cls(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )
        
    @classmethod
    def from_retinanet(cls, retinanet_results, conf_threshold: int=0.5) -> Self:
        boxes, confidences, class_ids = retinanet_results[0]
        mask = confidences > conf_threshold
        
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        meta = retinanet_results[1]
        boxes = scale_boxes(boxes=boxes, meta=meta)
        
        return cls(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            np.ndarray,
            np.ndarray | None,
            float | None,
            int | None,
            int | None,
            dict[str, np.ndarray | list],
        ]
    ]:
        """
        Iterates over the Detections object and yield a tuple of
        `(xyxy, mask, confidence, class_id, tracker_id, data)` for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.mask[i] if self.mask is not None else None,
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
                get_data_item(self.data, i),
            )

    def __eq__(self, other: Self):
        return all(
            [
                np.array_equal(self.xyxy, other.xyxy),
                np.array_equal(self.mask, other.mask),
                np.array_equal(self.class_id, other.class_id),
                np.array_equal(self.confidence, other.confidence),
                np.array_equal(self.tracker_id, other.tracker_id),
                is_data_equal(self.data, other.data),
                is_metadata_equal(self.metadata, other.metadata),
            ]
        )
    
    
    def __getitem__(
        self, index: int | slice | list[int] | np.ndarray | str
    ) -> Self | list | np.ndarray | None:
        """
        Get a subset of the Detections object or access an item from its data field.

        When provided with an integer, slice, list of integers, or a numpy array, this
        method returns a new Detections object that represents a subset of the original
        detections. When provided with a string, it accesses the corresponding item in
        the data dictionary.

        Args:
            index (Union[int, slice, List[int], np.ndarray, str]): The index, indices,
                or key to access a subset of the Detections or an item from the data.

        Returns:
            Union[Detections, Any]: A subset of the Detections object or an item from
                the data field.

        Example:
            ```python
            import swatahvision as sv

            detections = sv.Detections()

            first_detection = detections[0]
            first_10_detections = detections[0:10]
            some_detections = detections[[0, 2, 4]]
            class_0_detections = detections[detections.class_id == 0]
            high_confidence_detections = detections[detections.confidence > 0.5]

            feature_vector = detections['feature_vector']
            ```
        """
        if isinstance(index, str):
            return self.data.get(index)
        if self.is_empty():
            return self
        if isinstance(index, int):
            index = [index]
        return Detections(
            xyxy=self.xyxy[index],
            mask=self.mask[index] if self.mask is not None else None,
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            tracker_id=self.tracker_id[index] if self.tracker_id is not None else None,
            data=get_data_item(self.data, index),
            metadata=self.metadata,
        )

    
    def __setitem__(self, key: str, value: np.ndarray | list):
        """
        Set a value in the data dictionary of the Detections object.

        Args:
            key (str): The key in the data dictionary to set.
            value (Union[np.ndarray, List]): The value to set for the key.

        Example:
            ```python
            import cv2
            import swatahvision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s.pt')

            result = model(image)[0]
            detections = sv.Detections.from_ultralytics(result)

            detections['names'] = [
                 model.model.names[class_id]
                 for class_id
                 in detections.class_id
             ]
            ```
        """
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Value must be a np.ndarray or a list")

        if isinstance(value, list):
            value = np.array(value)

        self.data[key] = value
    
        
    @classmethod
    def empty(cls) -> Self:
        """
        Create an empty Detections object with no bounding boxes,
            confidences, or class IDs.

        Returns:
            (Detections): An empty Detections object.

        Example:
            ```python
            from swatahvision import Detections

            empty_detections = Detections.empty()
            ```
        """
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )
        
    def is_empty(self) -> bool:
        """
        Returns `True` if the `Detections` object is considered empty.
        """
        empty_detections = Detections.empty()
        empty_detections.data = self.data
        empty_detections.metadata = self.metadata
        return self == empty_detections

    @classmethod
    def merge(cls, detections_list: list[Self]) -> Self:
        """
        Merge a list of Detections objects into a single Detections object.

        This method takes a list of Detections objects and combines their
        respective fields (`xyxy`, `mask`, `confidence`, `class_id`, and `tracker_id`)
        into a single Detections object.

        For example, if merging Detections with 3 and 4 detected objects, this method
        will return a Detections with 7 objects (7 entries in `xyxy`, `mask`, etc).

        !!! Note

            When merging, empty `Detections` objects are ignored.

        Args:
            detections_list (List[Detections]): A list of Detections objects to merge.

        Returns:
            (Detections): A single Detections object containing
                the merged data from the input list.

        Example:
            ```python
            import numpy as np
            import swatahvision as sv

            detections_1 = sv.Detections(
                xyxy=np.array([[15, 15, 100, 100], [200, 200, 300, 300]]),
                class_id=np.array([1, 2]),
                data={'feature_vector': np.array([0.1, 0.2])}
            )

            detections_2 = sv.Detections(
                xyxy=np.array([[30, 30, 120, 120]]),
                class_id=np.array([1]),
                data={'feature_vector': np.array([0.3])}
            )

            merged_detections = sv.Detections.merge([detections_1, detections_2])

            merged_detections.xyxy
            array([[ 15,  15, 100, 100],
                   [200, 200, 300, 300],
                   [ 30,  30, 120, 120]])

            merged_detections.class_id
            array([1, 2, 1])

            merged_detections.data['feature_vector']
            array([0.1, 0.2, 0.3])
            ```
        """
        detections_list = [
            detections for detections in detections_list if not detections.is_empty()
        ]

        if len(detections_list) == 0:
            return Detections.empty()

        for detections in detections_list:
            validate_detections_fields(
                xyxy=detections.xyxy,
                mask=detections.mask,
                confidence=detections.confidence,
                class_id=detections.class_id,
                tracker_id=detections.tracker_id,
                data=detections.data,
            )

        xyxy = np.vstack([d.xyxy for d in detections_list])

        def stack_or_none(name: str):
            if all(d.__getattribute__(name) is None for d in detections_list):
                return None
            if any(d.__getattribute__(name) is None for d in detections_list):
                raise ValueError(f"All or none of the '{name}' fields must be None")
            return (
                np.vstack([d.__getattribute__(name) for d in detections_list])
                if name == "mask"
                else np.hstack([d.__getattribute__(name) for d in detections_list])
            )

        mask = stack_or_none("mask")
        confidence = stack_or_none("confidence")
        class_id = stack_or_none("class_id")
        tracker_id = stack_or_none("tracker_id")

        data = merge_data([d.data for d in detections_list])

        metadata_list = [detections.metadata for detections in detections_list]
        metadata = merge_metadata(metadata_list)

        return cls(
            xyxy=xyxy,
            mask=mask,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data=data,
            metadata=metadata,
        )


    def get_anchors_coordinates(self, anchor: Position) -> np.ndarray:
        """
        Calculates and returns the coordinates of a specific anchor point
        within the bounding boxes defined by the `xyxy` attribute. The anchor
        point can be any of the predefined positions in the `Position` enum,
        such as `CENTER`, `CENTER_LEFT`, `BOTTOM_RIGHT`, etc.

        Args:
            anchor (Position): An enum specifying the position of the anchor point
                within the bounding box. Supported positions are defined in the
                `Position` enum.

        Returns:
            np.ndarray: An array of shape `(n, 2)`, where `n` is the number of bounding
                boxes. Each row contains the `[x, y]` coordinates of the specified
                anchor point for the corresponding bounding box.

        Raises:
            ValueError: If the provided `anchor` is not supported.
        """
        if anchor == Position.CENTER:
            return np.array(
                [
                    (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2,
                    (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                ]
            ).transpose()
        elif anchor == Position.CENTER_OF_MASS:
            if self.mask is None:
                raise ValueError(
                    "Cannot use `Position.CENTER_OF_MASS` without a detection mask."
                )
            return calculate_masks_centroids(masks=self.mask)
        elif anchor == Position.CENTER_LEFT:
            return np.array(
                [
                    self.xyxy[:, 0],
                    (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                ]
            ).transpose()
        elif anchor == Position.CENTER_RIGHT:
            return np.array(
                [
                    self.xyxy[:, 2],
                    (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                ]
            ).transpose()
        elif anchor == Position.BOTTOM_CENTER:
            return np.array(
                [(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 3]]
            ).transpose()
        elif anchor == Position.BOTTOM_LEFT:
            return np.array([self.xyxy[:, 0], self.xyxy[:, 3]]).transpose()
        elif anchor == Position.BOTTOM_RIGHT:
            return np.array([self.xyxy[:, 2], self.xyxy[:, 3]]).transpose()
        elif anchor == Position.TOP_CENTER:
            return np.array(
                [(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 1]]
            ).transpose()
        elif anchor == Position.TOP_LEFT:
            return np.array([self.xyxy[:, 0], self.xyxy[:, 1]]).transpose()
        elif anchor == Position.TOP_RIGHT:
            return np.array([self.xyxy[:, 2], self.xyxy[:, 1]]).transpose()

        raise ValueError(f"{anchor} is not supported.")
    
    def with_nms(
        self,
        threshold: float = 0.5,
        class_agnostic: bool = False,
        overlap_metric: OverlapMetric = OverlapMetric.IOU,
    ) -> Self:
        """
        Performs non-max suppression on detection set. If the detections result
        from a segmentation model, the IoU mask is applied. Otherwise, box IoU is used.

        Args:
            threshold (float): The intersection-over-union threshold
                to use for non-maximum suppression. I'm the lower the value the more
                restrictive the NMS becomes. Defaults to 0.5.
            class_agnostic (bool): Whether to perform class-agnostic
                non-maximum suppression. If True, the class_id of each detection
                will be ignored. Defaults to False.
            overlap_metric (OverlapMetric): Metric used to compute the degree of
                overlap between pairs of masks or boxes (e.g., IoU, IoS).

        Returns:
            Detections: A new Detections object containing the subset of detections
                after non-maximum suppression.

        Raises:
            AssertionError: If `confidence` is None and class_agnostic is False.
                If `class_id` is None and class_agnostic is False.
        """
        if len(self) == 0:
            return self

        assert self.confidence is not None, (
            "Detections confidence must be given for NMS to be executed."
        )

        if class_agnostic:
            predictions = np.hstack((self.xyxy, self.confidence.reshape(-1, 1)))
        else:
            assert self.class_id is not None, (
                "Detections class_id must be given for NMS to be executed. If you"
                " intended to perform class agnostic NMS set class_agnostic=True."
            )
            predictions = np.hstack(
                (
                    self.xyxy,
                    self.confidence.reshape(-1, 1),
                    self.class_id.reshape(-1, 1),
                )
            )

        if self.mask is not None:
            indices = mask_non_max_suppression(
                predictions=predictions,
                masks=self.mask,
                iou_threshold=threshold,
                overlap_metric=overlap_metric,
            )
        else:
            indices = box_non_max_suppression(
                predictions=predictions,
                iou_threshold=threshold,
                overlap_metric=overlap_metric,
            )

        return self[indices]
    
    def with_nmm(
        self,
        threshold: float = 0.5,
        class_agnostic: bool = False,
        overlap_metric: OverlapMetric = OverlapMetric.IOU,
    ) -> Self:
        """
        Perform non-maximum merging on the current set of object detections.

        Args:
            threshold (float): The intersection-over-union threshold
                to use for non-maximum merging. Defaults to 0.5.
            class_agnostic (bool): Whether to perform class-agnostic
                non-maximum merging. If True, the class_id of each detection
                will be ignored. Defaults to False.
            overlap_metric (OverlapMetric): Metric used to compute the degree of
                overlap between pairs of masks or boxes (e.g., IoU, IoS).

        Returns:
            Detections: A new Detections object containing the subset of detections
                after non-maximum merging.

        Raises:
            AssertionError: If `confidence` is None or `class_id` is None and
                class_agnostic is False.

        ![non-max-merging](https://media.roboflow.com/supervision-docs/non-max-merging.png){ align=center width="800" }
        """  # noqa: E501 // docs
        if len(self) == 0:
            return self

        assert self.confidence is not None, (
            "Detections confidence must be given for NMM to be executed."
        )

        if class_agnostic:
            predictions = np.hstack((self.xyxy, self.confidence.reshape(-1, 1)))
        else:
            assert self.class_id is not None, (
                "Detections class_id must be given for NMM to be executed. If you"
                " intended to perform class agnostic NMM set class_agnostic=True."
            )
            predictions = np.hstack(
                (
                    self.xyxy,
                    self.confidence.reshape(-1, 1),
                    self.class_id.reshape(-1, 1),
                )
            )

        if self.mask is not None:
            merge_groups = mask_non_max_merge(
                predictions=predictions,
                masks=self.mask,
                iou_threshold=threshold,
                overlap_metric=overlap_metric,
            )
        else:
            merge_groups = box_non_max_merge(
                predictions=predictions,
                iou_threshold=threshold,
                overlap_metric=overlap_metric,
            )

        result = []
        for merge_group in merge_groups:
            unmerged_detections = [self[i] for i in merge_group]
            merged_detections = merge_inner_detections_objects_without_iou(
                unmerged_detections
            )
            result.append(merged_detections)

        return Detections.merge(result)   
    
def merge_inner_detection_object_pair(
    detections_1: Detections, detections_2: Detections
) -> Detections:
    """
    Merges two Detections object into a single Detections object.
    Assumes each Detections contains exactly one object.

    A `winning` detection is determined based on the confidence score of the two
    input detections. This winning detection is then used to specify which
    `class_id`, `tracker_id`, and `data` to include in the merged Detections object.

    The resulting `confidence` of the merged object is calculated by the weighted
    contribution of ea detection to the merged object.
    The bounding boxes and masks of the two input detections are merged into a
    single bounding box and mask, respectively.

    Args:
        detections_1 (Detections):
            The first Detections object
        detections_2 (Detections):
            The second Detections object

    Returns:
        Detections: A new Detections object, with merged attributes.

    Raises:
        ValueError: If the input Detections objects do not have exactly 1 detected
            object.

    Example:
        ```python
        import cv2
        import swatahvision as sv
        from inference import get_model

        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        model = get_model(model_id="yolov8s-640")

        result = model.infer(image)[0]
        detections = sv.Detections.from_inference(result)

        merged_detections = merge_object_detection_pair(
            detections[0], detections[1])
        ```
    """
    if len(detections_1) != 1 or len(detections_2) != 1:
        raise ValueError("Both Detections should have exactly 1 detected object.")

    validate_fields_both_defined_or_none(detections_1, detections_2)

    xyxy_1 = detections_1.xyxy[0]
    xyxy_2 = detections_2.xyxy[0]
    if detections_1.confidence is None and detections_2.confidence is None:
        merged_confidence = None
    else:
        detection_1_area = (xyxy_1[2] - xyxy_1[0]) * (xyxy_1[3] - xyxy_1[1])
        detections_2_area = (xyxy_2[2] - xyxy_2[0]) * (xyxy_2[3] - xyxy_2[1])
        merged_confidence = (
            detection_1_area * detections_1.confidence[0]
            + detections_2_area * detections_2.confidence[0]
        ) / (detection_1_area + detections_2_area)
        merged_confidence = np.array([merged_confidence])

    merged_x1, merged_y1 = np.minimum(xyxy_1[:2], xyxy_2[:2])
    merged_x2, merged_y2 = np.maximum(xyxy_1[2:], xyxy_2[2:])
    merged_xyxy = np.array([[merged_x1, merged_y1, merged_x2, merged_y2]])

    if detections_1.mask is None and detections_2.mask is None:
        merged_mask = None
    else:
        merged_mask = np.logical_or(detections_1.mask, detections_2.mask)

    if detections_1.confidence is None and detections_2.confidence is None:
        winning_detection = detections_1
    elif detections_1.confidence[0] >= detections_2.confidence[0]:
        winning_detection = detections_1
    else:
        winning_detection = detections_2

    metadata = merge_metadata([detections_1.metadata, detections_2.metadata])

    return Detections(
        xyxy=merged_xyxy,
        mask=merged_mask,
        confidence=merged_confidence,
        class_id=winning_detection.class_id,
        tracker_id=winning_detection.tracker_id,
        data=winning_detection.data,
        metadata=metadata,
    )


def merge_inner_detections_objects(
    detections: list[Detections],
    threshold=0.5,
    overlap_metric: OverlapMetric = OverlapMetric.IOU,
) -> Detections:
    """
    Given N detections each of length 1 (exactly one object inside), combine them into a
    single detection object of length 1. The contained inner object will be the merged
    result of all the input detections.

    For example, this lets you merge N boxes into one big box, N masks into one mask,
    etc.
    """
    detections_1 = detections[0]
    for detections_2 in detections[1:]:
        if detections_1.mask is not None and detections_2.mask is not None:
            iou = mask_iou_batch(detections_1.mask, detections_2.mask, overlap_metric)[
                0
            ]
        else:
            iou = box_iou_batch(detections_1.xyxy, detections_2.xyxy, overlap_metric)[0]
        if iou < threshold:
            break
        detections_1 = merge_inner_detection_object_pair(detections_1, detections_2)
    return detections_1


def merge_inner_detections_objects_without_iou(
    detections: list[Detections],
) -> Detections:
    """
    Given N detections each of length 1 (exactly one object inside), combine them into a
    single detection object of length 1. The contained inner object will be the merged
    result of all the input detections.

    For example, this lets you merge N boxes into one big box, N masks into one mask,
    etc.
    """
    return reduce(merge_inner_detection_object_pair, detections)


def validate_fields_both_defined_or_none(
    detections_1: Detections, detections_2: Detections
) -> None:
    """
    Verify that for each optional field in the Detections, both instances either have
    the field set to None or both have it set to non-None values.

    `data` field is ignored.

    Raises:
        ValueError: If one field is None and the other is not, for any of the fields.
    """
    attributes = get_instance_variables(detections_1)
    for attribute in attributes:
        value_1 = getattr(detections_1, attribute)
        value_2 = getattr(detections_2, attribute)

        if (value_1 is None) != (value_2 is None):
            raise ValueError(
                f"Field '{attribute}' should be consistently None or not None in both "
                "Detections."
            )
