from swatahvision.model.resolver import ModelResolver
from swatahvision.constraints import Engine, Hardware
from typing import Union


class Model:
    def __init__(self, model: str, engine: Engine=Engine.ONNX, hardware: Hardware=Hardware.CPU):
        model_resolver =  ModelResolver()
        self.runtime_engine = model_resolver.resolve(model=model, engine=engine, hardware=hardware)

    def __call__(self, input_image, input_size: Union[int, tuple[int, int]] = None):
        return self.runtime_engine.infer(input_image, input_size)