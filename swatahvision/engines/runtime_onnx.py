from typing import Union
from swatahvision.engines.base import RuntimeEngine
from swatahvision.constraints import Hardware
import onnxruntime as ort
import numpy as np
import cv2

class OnnxRuntimeEngine(RuntimeEngine):
    def infer(self, input_image, input_size: Union[int, tuple[int, int]] = None):
        
        # preprocessing can be added here
        input0_dtype = self.input_dtypes[self.input_names[0]]
        
        if input_size is None:
            input_size = tuple(self.input_shapes[self.input_names[0]])

        input_image, meta = self.preprocess(input_image, input0_dtype, input_size)

        raw_output = self.session.run(self.output_names, {self.input_names[0]: input_image})
        return raw_output, meta

    def load(self, model_path: str, hardware: Hardware = Hardware.CPU):
        provider = {Hardware.GPU: "CUDAExecutionProvider", Hardware.CPU: "CPUExecutionProvider"}[hardware]
        self.session = ort.InferenceSession(
        model_path,
        providers=[provider]
        )
        
        self.input_names, self.input_shapes, self.input_dtypes, self.output_names = self.get_model_info(self.session)

    def get_model_info(cls, session):
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

        for inp in session.get_inputs():
            input_names.append(inp.name)
            input_shapes[inp.name] = inp.shape
            input_dtypes[inp.name] = inp.type

        output_names = [out.name for out in session.get_outputs()]
        
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