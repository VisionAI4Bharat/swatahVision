import swatahvision as sv
import numpy as np

# ---------------------------------------------
# Load MoveNet model
# - OpenVINO inference engine
# - Running on CPU
# ---------------------------------------------
model = sv.Model(
    model="movenet.xml",  # Replace with your OpenVINO IR XML file
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)

# ---------------------------------------------
# Load input image from file
# ---------------------------------------------
image = sv.Image.load_from_file("assets/sample.jpg")  # Replace with your image

# ---------------------------------------------
# Run pose estimation inference
# ---------------------------------------------
outs = model(image)

# ---------------------------------------------
# Parse MoveNet output
# - Expected shapes: (1,1,17,3) or (1,17,3)
# - Each keypoint: y, x, confidence
# ---------------------------------------------
output = outs
while isinstance(output, (list, tuple)):
    output = output[0]

output = np.asarray(output)

if output.ndim == 4:
    kps = output[0][0]
elif output.ndim == 3:
    kps = output[0]
else:
    raise ValueError(f"Invalid MoveNet output shape: {output.shape}")

# Convert to (x, y) + confidence
keypoints = np.stack([kps[:, 1], kps[:, 0]], axis=1)  # x, y
confidence = kps[:, 2]

# ---------------------------------------------
# Print keypoints and confidence
# ---------------------------------------------
for i, (kp, conf) in enumerate(zip(keypoints, confidence)):
    print(f"Keypoint {i}: x={kp[0]:.2f}, y={kp[1]:.2f}, confidence={conf:.2f}")