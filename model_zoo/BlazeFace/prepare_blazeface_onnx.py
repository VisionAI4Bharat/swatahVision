import onnx
from onnx import helper

# Load original model
model = onnx.load("blaze.onnx")

# Create constant tensors
conf_tensor = helper.make_tensor(
    name="conf_threshold",
    data_type=onnx.TensorProto.FLOAT,
    dims=[1],
    vals=[0.5],
)

max_det_tensor = helper.make_tensor(
    name="max_detections",
    data_type=onnx.TensorProto.INT64,
    dims=[1],
    vals=[25],
)

iou_tensor = helper.make_tensor(
    name="iou_threshold",
    data_type=onnx.TensorProto.FLOAT,
    dims=[1],
    vals=[0.3],
)

# Add constants to model
model.graph.initializer.extend([
    conf_tensor,
    max_det_tensor,
    iou_tensor
])

# Keep only image input
image_input = model.graph.input[0]
model.graph.ClearField("input")
model.graph.input.extend([image_input])

# Save fixed model
onnx.save(model, "blaze_fixed.onnx")

print("Saved fixed model → blaze_fixed.onnx")