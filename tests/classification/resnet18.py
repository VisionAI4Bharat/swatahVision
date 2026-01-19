import swatahvision as sv

# ---------------------------------------------
# Load ResNet-18 classification model
# - ONNX inference engine
# - Running on CPU
# ---------------------------------------------
model = sv.Model(model="resnet18-13", engine=sv.Engine.ONNX, hardware=sv.Hardware.CPU)

# ---------------------------------------------
# Load input image from file
# ---------------------------------------------
image = sv.Image.load_from_file("assets/car.jpg")

# ---------------------------------------------
# Run image classification inference
# ---------------------------------------------
outs = model(image)

# ---------------------------------------------
# Convert raw model output to classification results
# Get top-5 predicted classes
# ---------------------------------------------
classification = sv.Classification.from_resnet(outs, top_k=5)

# ---------------------------------------------
# Print classification results
# ---------------------------------------------
print(classification)
