import swatahVision as sv

# ---------------------------------------------
# Load BlazeFace face detection model
# ---------------------------------------------
model = sv.Model(
    model="C:\\Users\\LENOVO\\Downloads\\blaze_fixed.onnx",
    engine=sv.Engine.ONNX,
    hardware=sv.Hardware.CPU
)

# ---------------------------------------------
# Load input image
# ---------------------------------------------
image = sv.Image.load_from_file("C:\\Users\\LENOVO\\Downloads\\example1.png")

# ---------------------------------------------
# Run inference
# ---------------------------------------------
outs = model(image)

# ---------------------------------------------
# Print model outputs
# ---------------------------------------------
print("Model outputs:", outs)