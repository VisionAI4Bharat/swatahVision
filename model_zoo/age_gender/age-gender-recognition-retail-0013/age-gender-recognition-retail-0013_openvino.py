import swatahvision as sv


model = sv.Model(
    model="age-gender-recognition-retail-0013.xml",
    engine=sv.Engine.OPENVINO,
    hardware=sv.Hardware.CPU
)


image = sv.Image.load_from_file("assets/face.jpeg")

# ---------------------------------------------
# Run inference
# ---------------------------------------------
outs = model(image)

# ---------------------------------------------
# Decode outputs manually
# ---------------------------------------------
gender_tensor = outs[0][0]
age_tensor = outs[0][1]

age = int(age_tensor[0][0][0][0] * 100)
gender = "Male" if gender_tensor[0][1][0][0] > gender_tensor[0][0][0][0] else "Female"


print("Predicted Age:", age)
print("Predicted Gender:", gender)