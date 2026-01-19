import cv2
import numpy as np
from swatahvision.draw.image import Image
from swatahvision.core.classification.core import Classification
from swatahvision.core.detection.core import Detections

class UI():
    @staticmethod
    def draw_classification_labels(
        image:Image,
        classification:Classification,
        start_x:int=10,
        start_y:int=25,
        line_gap:int=22,
        font_scale:float=0.6,
        thickness:int=1
    ):
        """
        Draws classification confidence on the top-left corner of an image.
        """
        
        class_ids = classification.class_id
        confidences = classification.confidence
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (cls_id, conf) in enumerate(zip(class_ids[0], confidences[0])):
            text = f"class: {cls_id} {conf:.2f}"
            y = start_y + i * line_gap

            cv2.putText(
                image,
                text,
                (start_x, y),
                font,
                font_scale,
                (0, 255, 0),  # Green text
                thickness,
                cv2.LINE_AA
            )

        return image
    
    @staticmethod
    def draw_bboxes(
        image:Image, 
        detections:Detections, 
        conf:float=0.5, 
        font_scale:float=0.6, 
        thickness:int=1
        ):
        boxes = detections.xyxy
        scores = detections.confidence
        class_ids = detections.class_id

        keep_idx = np.where(scores >= conf)[0]

        filtered_boxes = boxes[keep_idx]
        filtered_scores = scores[keep_idx]
        filtered_class_ids = class_ids[keep_idx]
    
        for box, score, class_id in zip(filtered_boxes, filtered_scores, filtered_class_ids):
            x1, y1, x2, y2 = map(int, box)
    
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                thickness
            )

            label = f"class: {class_id} {score:.2f} "
            cv2.putText(
                image,
                label,
                (x1, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA
            )

        return image