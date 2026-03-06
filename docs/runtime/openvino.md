# OpenVINO Runtime Engine for SwatahVision

This code creates a **custom runtime engine** that allows **OpenVINO models** to run inside the **SwatahVision framework**.

In simple terms, this code helps the system:

1. Load an OpenVINO model
2. Prepare an image so the model can understand it
3. Run the model on the image
4. Return the prediction results

---

# What is OpenVINO?

**OpenVINO** is a toolkit developed by Intel that helps run AI models **faster and more efficiently**, especially on:

- CPUs
- GPUs
- Intel hardware

It is commonly used for **computer vision tasks** such as:

- Image classification
- Object detection
- Face recognition

---

# Purpose of This Code

This file creates a class called:

```
OpenVinoRuntimeEngine
```

This class connects:

```
SwatahVision → OpenVINO Runtime
```

This allows SwatahVision to **run OpenVINO models easily**.

---

# Libraries Used

This code uses the following Python libraries:

- **openvino** → runs the OpenVINO model
- **numpy** → handles numerical data
- **opencv (cv2)** → processes images

---

# Class: OpenVinoRuntimeEngine

This class extends the base class:

```
RuntimeEngine
```

It defines how the system should:

- load a model
- prepare the input image
- run the model
- return the results

---

# 1. Loading the Model

Function:

```
load()
```

This function loads the OpenVINO model into memory.

Steps performed:

1. Select the hardware (CPU or GPU)
2. Create an OpenVINO Core object
3. Read the model file
4. Compile the model for the selected device

Example hardware options:

```
CPU → runs on processor
GPU → runs on graphics card
```

After loading the model, the function also collects information about:

- input names
- input shapes
- input data types
- output names

---

# 2. Running the Model (Inference)

Function:

```
infer()
```

This function performs the **actual prediction**.

Steps:

1. Receive an input image
2. Prepare the image using preprocessing
3. Send the image to the compiled model
4. Get the prediction results

The function returns:

```
raw_output
meta_information
```

---

# 3. Getting Model Information

Function:

```
get_model_info()
```

This function extracts important details from the model.

It collects:

- Input names
- Input shapes
- Input data types
- Output names

This information helps the engine understand **what format the model expects**.

---

# 4. Image Preprocessing

Function:

```
preprocess()
```

Before sending an image to the model, the image must be prepared correctly.

This function performs several steps to prepare the image.

---

## Step 1: Resize the Image

The image is resized to match the size expected by the model.

---

## Step 2: Keep Image Proportions (Letterbox)

Instead of stretching the image, the code keeps the original shape.

This is done using a technique called **letterboxing**.

The process:

1. Resize the image
2. Add padding around the image

This prevents distortion.

---

## Step 3: Rearrange Image Format

Normally images are stored as:

```
(H, W, C)
Height, Width, Channels
```

But most AI models expect:

```
(C, H, W)
Channels, Height, Width
```

So the code rearranges the image format.

---

## Step 4: Handle Single or Multiple Images

The engine can process:

- a **single image**
- **multiple images at once**

If needed, the code automatically adds a **batch dimension**.

---

## Step 5: Convert Image Data Type

Different models expect different input types such as:

- `float32`
- `uint8`

The code automatically converts the image to the correct type.

---

# Letterbox Function

The `letterbox()` function resizes an image **without stretching it**.

Steps performed:

1. Calculate how much the image should scale
2. Resize the image
3. Add padding to match the required size

Example:

Original image:

```
400 × 300
```

Model input size:

```
640 × 640
```

The code:

- resizes the image
- adds padding around it

This keeps the image looking natural.

---

# Metadata Returned

The preprocessing function also returns **metadata information**.

This includes:

```
scale
padding_x
padding_y
```

This data can later be used to adjust predictions back to the original image size.

---

# Overall Workflow

The full process works like this:

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

# Why This Code is Useful

This engine makes it easier to run OpenVINO models in SwatahVision.

It provides:

- automatic image preprocessing
- support for CPU and GPU
- support for batch inputs
- simple model execution

---

# Summary

This code acts as a bridge between:

```
SwatahVision Framework
        ↓
OpenVINO Runtime Engine
        ↓
AI Model
```

It simplifies the process of:

- loading models
- preparing images
- running predictions
- retrieving results

This makes it easier to build **computer vision applications using OpenVINO models**.