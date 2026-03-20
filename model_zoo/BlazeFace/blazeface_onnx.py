import swatahVision as sv

# ---------------------------------------------
# Load BlazeFace face detection model
# ---------------------------------------------
model = sv.Model(
    model="blazeface.onnx",
    engine=sv.Engine.ONNX,
    hardware=sv.Hardware.CPU
)

# ---------------------------------------------
# Load input image
# ---------------------------------------------
image = sv.Image.load_from_file("assets/sample.jpg")

# ---------------------------------------------
# Run inference
# ---------------------------------------------
outs = model(image)

# ---------------------------------------------
# Print model outputs
# ---------------------------------------------
print("Model outputs:", outs)