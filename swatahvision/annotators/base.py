from abc import ABC, abstractmethod

from swatahVision.core.detection.core import Detections
from swatahVision.draw.base import ImageType 


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:
        pass