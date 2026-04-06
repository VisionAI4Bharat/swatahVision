from abc import ABC, abstractmethod

from swatahvision.core.detection.core import Detections
from swatahvision.draw.base import ImageType 


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:
        pass