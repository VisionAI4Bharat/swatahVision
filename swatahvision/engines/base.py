from abc import ABC, abstractmethod
from typing import Union

class RuntimeEngine(ABC):
    @abstractmethod
    def load(self, model_path: str):
        pass
    @abstractmethod
    def infer(self, input_image):
        pass
    @abstractmethod
    def preprocess(self, input_image, input_type, input_shape: Union[int | tuple[int, int]]):
        pass
    @abstractmethod
    def get_model_info(self, model):
        pass