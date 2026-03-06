# ONNX Runtime Engine for SwatahVision

This code creates a **custom inference engine** that allows models in **ONNX format** to run inside the **SwatahVision framework**.

In simple terms, it helps the system:

1. Load an ONNX model
2. Prepare an image so the model can understand it
3. Run the model on the image
4. Return the prediction results

---

# What is ONNX?

**ONNX (Open Neural Network Exchange)** is a format used to store machine learning models.

It allows models created in frameworks like:

- PyTorch
- TensorFlow
- Keras

to run in many different environments.

This code uses **ONNX Runtime**, which is a tool that executes ONNX models efficiently.

---

# Main Purpose of This Code

This file creates a class called:

```
OnnxRuntimeEngine
```

This class connects:

**SwatahVision → ONNX Runtime**

So that SwatahVision can run ONNX models easily.

---

# Libraries Used

The code uses the following Python libraries:

- **onnxruntime** → runs ONNX models
- **numpy** → handles numerical data
- **opencv (cv2)** → processes images

---

# Class: OnnxRuntimeEngine

This class inherits from the base engine:

```
RuntimeEngine
```

It defines how:

- the model is loaded
- the input image is prepared
- the model prediction is executed

---

# 1. Loading the Model

Function:

```
load()
```

This function loads the ONNX model into memory.

It also selects the hardware used to run the model.

Possible hardware options:

- **CPU**
- **GPU**

Example behavior:

```
CPU → CPUExecutionProvider
GPU → CUDAExecutionProvider
```

After loading the model, the function also collects information about:

- model inputs
- input shapes
- input data types
- model outputs

---

# 2. Running the Model (Inference)

Function:

```
infer()
```

This function performs the main prediction.

Steps performed:

1. Take the input image
2. Prepare the image using preprocessing
3. Send the image to the model
4. Get the prediction result

The model then returns:

- **raw output**
- **metadata information**

---

# 3. Getting Model Information

Function:

```
get_model_info()
```

This function reads details from the ONNX model.

It extracts information such as:

- Input names
- Input shapes
- Input data types
- Output names

This helps the engine understand **what format the model expects**.

---

# 4. Image Preprocessing

Function:

```
preprocess()
```

Before sending an image to the model, it must be prepared correctly.

This function performs that preparation.

The steps include:

### 1. Resize the image

The image is resized to match the model's expected input size.

---

### 2. Maintain aspect ratio (Letterbox)

Instead of stretching the image, the code keeps the original shape.

It does this by:

- resizing the image
- adding padding around it

This process is called **letterboxing**.

---

### 3. Change image format

Images normally look like this:

```
(H, W, C)
Height, Width, Channels
```

But models expect:

```
(C, H, W)
Channels, Height, Width
```

So the code rearranges the image format.

---

### 4. Handle batch inputs

The code can process:

- **single image**
- **multiple images at once**

If needed, it automatically adds a batch dimension.

---

### 5. Convert data type

The model may expect input as:

- `float32`
- `uint8`

The code converts the image to the correct type.

---

# Letterbox Function (Important)

The `letterbox()` function resizes an image **without distortion**.

Steps:

1. Calculate scaling factor
2. Resize image
3. Add padding to reach required size

Example:

Original Image

```
400 × 300
```

Model Input

```
640 × 640
```

The code:

- resizes the image
- adds black padding around it

This keeps the image proportions correct.

---

# Output of Preprocessing

The preprocessing function returns two things:

```
processed_image
meta_information
```

The metadata contains:

- scale factor
- padding values

This information can later be used to adjust predictions back to the original image size.

---

# Overall Workflow

The complete process looks like this:

```
Load Model
      ↓
Receive Image
      ↓
Preprocess Image
      ↓
Run Model
      ↓
Return Prediction
```

---

# Why This Code Is Useful

This engine allows:

- Running ONNX models inside SwatahVision
- Supporting both CPU and GPU
- Automatically handling image preprocessing
- Supporting single and batch inputs

It simplifies the process of **deploying ONNX models in computer vision applications**.

---

# Summary

This code builds a bridge between:

```
SwatahVision Framework
        ↓
ONNX Runtime Engine
        ↓
Machine Learning Model
```

It makes it easier to:

- load models
- prepare images
- run predictions
- retrieve results