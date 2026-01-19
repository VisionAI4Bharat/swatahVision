from swatahvision.engines.base import RuntimeEngine
from swatahvision.constraints import Engine, Hardware
from swatahvision.utils import file
from pathlib import Path

from swatahvision.engines.runtime_onnx import OnnxRuntimeEngine
from swatahvision.engines.runtime_openvino import OpenVinoRuntimeEngine

class ModelResolver():
    def resolve(self, model: str, engine: Engine=Engine.ONNX, hardware: Hardware=Hardware.CPU) -> RuntimeEngine:
       # Load model from path (model.onnx)
        if self._is_model_file(model):
            print("[info] Loading model from path:", model)
            return self._select_engine(model, engine, hardware)
        
        # Load model from hub / local cache
        if not self._is_model_file(model):
            print("[info] Loading model from hub (local):", model)
            model_file_path = self._get_model_file_path(model=model, engine=engine)
            
            if file.find_file_in_cache(model_file_path):
                print(f"[info] Loading model from cache: {model_file_path}")
                return self._select_engine(model_file_path, engine, hardware)
            else:
                # Download model from hub then select engine
                print(f"[info] Model not found in loacl cache")
    
    def _is_model_file(cls, model: str) -> bool:
        return Path(model).is_file()
    
    def _select_engine(cls, model_path: Path, engine: Engine, hardware: Hardware)-> RuntimeEngine:
        """
        Selects the appropriate engine based on model configuration and constraints.
        """
        if engine == Engine.ONNX:
            print("[info] Using ONNX Runtime Engine")
            runtime_engine = OnnxRuntimeEngine()
            runtime_engine.load(model_path=model_path, hardware=hardware)

            print(f"[info] Model loaded successfully")
            return runtime_engine
        
        if engine == Engine.OPENVINO:
            print("[info] Using OPENVINO Runtime Engine")
            runtime_engine = OpenVinoRuntimeEngine()
            runtime_engine.load(model_path=model_path, hardware=hardware)

            print(f"[info] Model loaded successfully")
            return runtime_engine
        
    def _get_model_file_path(cls, model: str, engine: Engine):
        folder_name, extention = cls._get_foldername_and_extention(engine=engine)
        model_file_path = file.get_cache_dir() / folder_name / f"{model}.{extention}"
        return model_file_path

    def _get_foldername_and_extention(cls, engine: Engine):
        if engine == Engine.ONNX:
            return engine.value, "onnx"
        if engine == Engine.OPENVINO:
            return engine.value, "xml"    
    