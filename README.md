# GPU Edge Detection with TensorFlow

## 🔍 Overview
This project demonstrates GPU-accelerated image processing using TensorFlow in Google Colab. It applies a Sobel edge detection filter on an input image (Ben 10 cartoon image) to extract horizontal edges using GPU hardware. This highlights the power of GPUs in performing convolution-based operations efficiently.

## ⚙️ Tools & Technologies Used
- **Google Colab** (GPU runtime - T4)
- **TensorFlow 2.18.0**
- **Python 3.11**
- **Matplotlib & PIL** for image visualization and saving

## 🧠 What This Project Does
- Loads a grayscale image (resized for efficiency)
- Defines a Sobel kernel (horizontal edges)
- Applies convolution via `tf.nn.conv2d()` using `/GPU:0`
- Saves and displays output image
- Confirms GPU usage via TensorFlow API

## 🖼️ Visual Results

| Original Image                          | Edge Detection (GPU Output)           |
|----------------------------------------|---------------------------------------|
| ![Original](./Ben TennysoN.jpg)        | ![Edge Output](./gpu_output_tf.png)   |

## 🚀 How to Run (In Google Colab)
1. Open the notebook `edge_detection_gpu_tensorflow.ipynb`
2. Go to `Runtime > Change Runtime Type` and select **GPU**
3. Upload any `.jpg` or `.png` image (Ben 10 used as example)
4. Run all cells
5. View and save the result image
6. Output will be `gpu_output_tf.png` and proof in `log.txt`

## 📂 Project Files

| File                          | Description                                      |
|-------------------------------|--------------------------------------------------|
| `edge_detection_gpu_tensorflow.ipynb` | Main Colab notebook with code                   |
| `gpu_output_tf.png`           | Output image from GPU-based Sobel filter        |
| `log.txt`                     | Text log confirming GPU usage                   |
| `Ben_Tennyson.jpg` (optional) | Input image used for demonstration              |
| `README.md`                   | This file                                        |

## ✅ GPU Verification
TensorFlow is used to detect available GPU devices and run edge detection under:
```python
with tf.device('/GPU:0'):
```

Output of:
```python
tf.config.list_physical_devices('GPU')
```
Should return:
```python
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 🎥 Presentation
A short demo video (5–10 minutes) is included separately to explain:
- What the code does
- How it uses GPU
- Visual comparison of input/output
- Lessons learned

## 📌 Conclusion
This project fulfills the GPU Specialization Capstone requirement by demonstrating a real-world GPU-accelerated application using TensorFlow. It's simple, visually meaningful, and easy to reproduce.

---
**Author:** [Sahil Tushir]  

