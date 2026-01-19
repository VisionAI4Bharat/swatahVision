from enum import Enum

class Hardware(Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    REMOTE = "remote"

class Engine(Enum):
    TFLITE = "tflite"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    RKNN = "rknn"
    OPENVINO = "openvino"
