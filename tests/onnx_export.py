import torch
import onnx
import onnxruntime as ort
import numpy as np

from torchvision.models import resnet18, ResNet18_Weights

EXPORT_MODEL_NAME = "resnet18-13.onnx"
INPUT_SHAPE = 224

# Load model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, INPUT_SHAPE, INPUT_SHAPE)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    EXPORT_MODEL_NAME,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
)

# Validate ONNX
onnx_model = onnx.load(EXPORT_MODEL_NAME)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")

# ONNX Runtime inference
session = ort.InferenceSession(EXPORT_MODEL_NAME)

image = np.random.randn(1, 3, INPUT_SHAPE, INPUT_SHAPE).astype(np.float32)
logits = session.run(None, {"input": image})[0]

print("Output shape:", logits.shape)
