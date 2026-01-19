from typing import Union
from swatahvision.engines.base import RuntimeEngine
from swatahvision.constraints import Hardware
import openvino as ov
import numpy as np
import cv2

class OpenVinoRuntimeEngine(RuntimeEngine):
    def infer(self, input_image, input_size: Union[int, tuple[int, int]] = None):
        
        # preprocessing can be added here
        input0_dtype = self.input_dtypes[self.input_names[0]]

        if input_size is None:
            input_size = tuple(self.input_shapes[self.input_names[0]])
            
        input_image, meta = self.preprocess(input_image, input0_dtype, input_size)
        
        outputs = self.compiled_model({self.input_names[0]: input_image})
        raw_output = [outputs[name] for name in self.output_names]
        return raw_output, meta

    def load(self, model_path: str, hardware: Hardware = Hardware.CPU):
        provider = {Hardware.GPU: "GPU", Hardware.CPU: "CPU"}[hardware]
        core = ov.Core()
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, provider)
        
        self.input_names, self.input_shapes, self.input_dtypes, self.output_names = self.get_model_info(self.compiled_model)
        
    def get_model_info(cls, compiled_model):
        """
        Returns:
            input_names   : List[str]
            input_shapes  : Dict[str, List[int | None]]
            input_dtypes  : Dict[str, np.dtype]
            output_names  : List[str]
        """
        input_names = []
        input_shapes = {}
        input_dtypes = {}

        for inp in compiled_model.inputs:
            name = inp.get_any_name()
            shape = [
                dim.get_length() if dim.is_static else -1
                for dim in inp.get_partial_shape()
            ]
            dtype = inp.get_element_type().get_type_name()

            input_names.append(name)
            input_shapes[name] = shape
            input_dtypes[name] = dtype

        output_names = []
        for out in compiled_model.outputs:
            output_names.append(out.get_any_name())
                
        return input_names, input_shapes, input_dtypes, output_names
    
    def preprocess(cls, input_image, input_type, input_shape: Union[int | tuple[int, int]]):
        
        height, width = input_shape[-2:]

        is_batched_input = input_image.ndim == 4  # (B, H, W, C)
        expects_batch = len(input_shape) == 4     # (B, C, H, W)
        
        def letterbox(img):
            h0, w0 = img.shape[:2]

            # ---- scale ----
            scale = min(width / w0, height / h0)
            new_w = int(w0 * scale)
            new_h = int(h0 * scale)

            # ---- resize ----
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # ---- padding ----
            pad_x = (width - new_w) // 2
            pad_y = (height - new_h) // 2

            padded = np.zeros((height, width, 3), dtype=resized.dtype)
            padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

            return padded, scale, pad_x, pad_y
        
        def process_single(img):
            img, scale, pad_x, pad_y = letterbox(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # CHW
            return img, scale, pad_x, pad_y
        
        # -------- Process --------
        if is_batched_input:
            outputs = [process_single(img) for img in input_image]
            processed = np.stack([o[0] for o in outputs], axis=0)
            meta = [(o[1], o[2], o[3]) for o in outputs]
        else:
            processed, scale, pad_x, pad_y = process_single(input_image)
            meta = (scale, pad_x, pad_y)
            if expects_batch:
                processed = np.expand_dims(processed, axis=0)
        
        # -------- Data type --------
        if "uint8" in input_type:
            processed = (processed * 255).astype(np.uint8)
        else:
            processed = processed.astype(np.float32)  
        
        return processed, meta