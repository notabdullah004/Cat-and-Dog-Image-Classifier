# Cat-and-Dog-Image-Classifier
Cat and Dog Image Classifier  Machine Learning with Python
# 🐱🐶 Cat and Dog Image Classifier

This project is part of the [freeCodeCamp Machine Learning with Python Certification](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/cat-and-dog-image-classifier). It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images as either a **cat** or a **dog**.

## 🚀 Features

- ✅ Trains a custom CNN on a filtered cats vs. dogs dataset
- ✅ Achieves ≥63% validation accuracy (required by FCC)
- ✅ Uses data augmentation to improve generalization
- ✅ Visualizes accuracy and loss over epochs
- ✅ Includes function to test on new images

---

## 📁 Dataset

The model is trained on the `cats_and_dogs_filtered` dataset provided by TensorFlow.

- 📦 Source: [Download link (hosted by TensorFlow)](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)
- 🐾 Classes: `cats`, `dogs`
- 🖼️ Image size: Resized to **150x150 pixels**

---

## 🧠 Model Architecture

The CNN architecture includes:

```plaintext
Input Layer (150x150x3)
↓
Conv2D (32 filters) + MaxPooling
↓
Conv2D (64 filters) + MaxPooling
↓
Conv2D (128 filters) + MaxPooling
↓
Conv2D (128 filters) + MaxPooling
↓
Flatten → Dropout (0.5) → Dense (512) → Output (sigmoid)
