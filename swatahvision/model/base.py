from swatahVision.model.resolver import ModelResolver
from swatahVision.constraints import Engine, Hardware
from typing import Union


class Model:
    def __init__(self, model: str, engine: Engine=Engine.ONNX, hardware: Hardware=Hardware.CPU):
        model_resolver =  ModelResolver()
        self.runtime_engine = model_resolver.resolve(model=model, engine=engine, hardware=hardware)
        
    def infer(self, input_image):
        return self.runtime_engine.infer(input_image)
        
    def __call__(self, input_image):
        return self.infer(input_image)