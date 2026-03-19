from typing import Union
from swatahvision.engines.base import RuntimeEngine
from swatahvision.constraints import Hardware
#import openvino as ov
import numpy as np
import cv2

class OpenVinoRuntimeEngine:
    def __init__(self, *args, **kwargs):
        pass

    def infer(self, input_image):
        return "Dummy output (OpenVINO disabled)"
    