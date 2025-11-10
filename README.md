# 🎯 Handwritten Digit Recognition using Deep Learning

**Framework:** TensorFlow / Keras  
**Dataset:** MNIST (Modified National Institute of Standards and Technology)

---

## 📌 Overview

This project is a **Deep Learning-based handwritten digit recognition system** that can accurately identify digits (**0–9**) from grayscale images.  
It uses the **MNIST dataset**, which consists of **70,000 images** of handwritten digits collected from diverse individuals.  

The project demonstrates two different architectures:
- 🧠 **Convolutional Neural Network (CNN)** – for high-accuracy image feature extraction  
- ⚙️ **Dense Neural Network (DNN)** – for understanding non-linear relationships in pixel data  

The goal is to explore and compare both approaches for digit classification while maintaining simplicity, speed, and accuracy.  
This project also includes **visualizations, evaluation metrics**, and **deployment readiness** (Streamlit/Flask).

---

## ✨ Key Features

- 🔢 **Digit Classification:** Accurately recognizes handwritten digits (0–9)  
- ⚡ **Dual Models:** Supports both CNN and Dense Neural Network architectures  
- 🚀 **Fast Training:** Trains in under 2 minutes on Google Colab GPU  
- 📊 **Comprehensive Evaluation:** Accuracy, confusion matrix, and learning curves  
- 🧩 **Modular Codebase:** Clean structure for easy understanding and modification  
- 🖼️ **Visualization:** Displays sample predictions using Matplotlib  
- 🌐 **Deployment Ready:** Easily integrable with Streamlit or Flask web apps  

---

## 🧠 Project Description

The **Handwritten Digit Recognition** project is a classic example of **image classification using Deep Learning**.  
It leverages the **MNIST dataset**, a gold standard benchmark dataset for computer vision research.  
The model takes an image of a handwritten digit (28×28 pixels) as input and predicts the corresponding digit (0–9) as output.

Two approaches are implemented and compared:
1. **Dense Neural Network (DNN)** – A fully connected feedforward neural network.
2. **Convolutional Neural Network (CNN)** – A more advanced architecture capable of extracting spatial and visual features from images.

By comparing these two models, we aim to understand how convolutional layers improve performance over simple dense layers in image-based tasks.

---

## 🗂️ Dataset Details

- **Name:** MNIST (Modified National Institute of Standards and Technology)  
- **Source:** [`tf.keras.datasets.mnist`](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist)  
- **Total Images:** 70,000  
  - 60,000 training images  
  - 10,000 testing images  
- **Image Dimensions:** 28 × 28 pixels (grayscale)  
- **Classes:** 10 (Digits 0–9)  
- **Format:** Each image is represented as a 28×28 matrix of pixel values (0–255).  

---

## 🎯 Objectives

- 📌 Build and train a Deep Learning model capable of classifying handwritten digits.  
- 🧩 Compare the performance of **Dense Neural Network** and **Convolutional Neural Network** architectures.  
- 📈 Evaluate model accuracy and visualize training progress using plots.  
- 🧠 Understand the effect of convolution and pooling layers in image recognition.  
- 🌐 Prepare the trained model for deployment using Streamlit or Flask.  

---

## 🛠️ Technology Stack

This project is built using a combination of **Deep Learning, Data Visualization, and Web Deployment** tools to ensure accuracy, interpretability, and usability.

| Category | Technologies / Tools |
|-----------|----------------------|
| **Programming Language** | Python 3.8+ |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Evaluation** | Scikit-learn |
| **Deployment (Optional)** | Streamlit / Flask |
| **Version Control** | Git, GitHub |
| **Environment** | Google Colab / Jupyter Notebook |

---

## ⚙️ Installation & Setup

Follow these steps to set up and run the project locally or on Google Colab:

### 🔹 Option 1: Run on Google Colab (Recommended)
1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Upload your project notebook (`Handwritten_Digit_Prediction.ipynb`).
3. Run all cells step-by-step.
4. Ensure the runtime type is set to **GPU** for faster training:  
   `Runtime → Change runtime type → GPU`.

---

### 🔹 Option 2: Run Locally on Your System

#### **1. Clone the Repository**
```bash
git clone https://github.com/Shivansh541/Handwritten-Digit-Recognition.git
cd handwritten-digit-recognition

```
#### **2. Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
venv\Scripts\activate  # for Windows
source venv/bin/activate  # for Mac/Linux
```
#### **3. Install Required Dependencies**
```bash
pip install -r requirements.txt
```
#### **4. Run the Script or Notebook**
```bash
jupyter notebook mnist_digit_recognition.ipynb
```

## 🧩 Model Architecture

This project implements and compares two Deep Learning models for handwritten digit recognition:

1. **Dense Neural Network (DNN)**
2. **Convolutional Neural Network (CNN)**

Both models are trained on the **MNIST dataset**, but differ in how they learn spatial and pixel-level patterns from the images.

---

### 🧠 1. Dense Neural Network (DNN)

#### ⚙️ Overview
A **Dense Neural Network**, also called a **Fully Connected Network**, connects every neuron from one layer to every neuron in the next layer.  
It is effective for learning global patterns in small datasets but less efficient for spatial data like images.

#### 🏗️ Architecture Design
| Layer | Type | Output Shape | Activation |
|--------|------|---------------|-------------|
| 1 | Flatten (28×28 → 784) | (784,) | — |
| 2 | Dense (128 units) | (128,) | ReLU |
| 3 | Dropout (0.2) | (128,) | — |
| 4 | Dense (64 units) | (64,) | ReLU |
| 5 | Dense (10 units) | (10,) | Softmax |

#### 🔍 Working Principle
- **Flatten Layer:** Converts each 28×28 image into a 1D vector of 784 values.  
- **Dense Layers:** Learn non-linear relationships between pixel intensities.  
- **Dropout:** Prevents overfitting by randomly disabling neurons during training.  
- **Output Layer:** Uses **Softmax** to output probability distribution for 10 classes (digits 0–9).

#### 🧮 Loss & Optimization
- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy  

#### 📊 Summary
The DNN performs well (≈97–98% accuracy) but struggles to capture **spatial features**, making it less robust to shifts or distortions in digits.

---

### 🧬 2. Convolutional Neural Network (CNN)

#### ⚙️ Overview
A **Convolutional Neural Network** is specifically designed for image data.  
It uses **convolutional filters** to extract spatial hierarchies — edges, shapes, and patterns — leading to higher accuracy and generalization.

#### 🏗️ Architecture Design
| Layer | Type | Output Shape | Activation |
|--------|------|---------------|-------------|
| 1 | Conv2D (32 filters, 3×3 kernel) | (26×26×32) | ReLU |
| 2 | MaxPooling2D (2×2) | (13×13×32) | — |
| 3 | Conv2D (64 filters, 3×3 kernel) | (11×11×64) | ReLU |
| 4 | MaxPooling2D (2×2) | (5×5×64) | — |
| 5 | Flatten | (1600,) | — |
| 6 | Dense (128 units) | (128,) | ReLU |
| 7 | Dropout (0.5) | (128,) | — |
| 8 | Dense (10 units) | (10,) | Softmax |

#### 🔍 Working Principle
- **Convolution Layers:** Extract features using filters that detect patterns (edges, lines, textures).  
- **Pooling Layers:** Reduce image dimensions while keeping essential information.  
- **Flatten Layer:** Converts feature maps into a single vector for classification.  
- **Dense Layers:** Combine learned features to make the final prediction.  

#### 🧮 Loss & Optimization
- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** Adam (Adaptive Moment Estimation)  
- **Metrics:** Accuracy  

#### ⚡ Performance
| Model | Training Time | Accuracy | Overfitting | Comments |
|--------|----------------|-----------|--------------|-----------|
| DNN | ~25 seconds | 97.8% | Moderate | Fast but limited feature learning |
| CNN | ~15 seconds | 99.2% | Low | Excellent accuracy and generalization |

---

### 🧠 Conceptual Difference

| Concept | DNN | CNN |
|----------|-----|-----|
| **Input Handling** | Flattened pixels (loses spatial info) | 2D structure preserved |
| **Feature Extraction** | Manual / learned by dense weights | Automatic via filters |
| **Overfitting** | More prone | Less prone |
| **Computation Time** | Faster | Slightly longer |
| **Accuracy** | High (~97%) | Very High (~99%) |
| **Best Used For** | Tabular or small image data | All image recognition tasks |

---

### 📉 Model Visualization (Example)
```python
model.summary()
```
## ANN Model Summary
| Layer (type)        | Output Shape | Param # |
| ------------------- | ------------ | ------- |
| **Flatten (flatten_2)** | (None, 784)  | 0       |
| **Dense (dense_5)**     | (None, 128)  | 100,480 |
| **Dropout (dropout_2)** | (None, 128)  | 0       |
| **Dense (dense_6)**     | (None, 64)   | 8,256   |
| **Dense (dense_7)**     | (None, 10)   | 650     |

**Total params:** 109,386 (427.29 KB)
**Trainable params:** 109,386 (427.29 KB)
**Non-trainable params:** 0 (0.00 B)

## CNN Model Summary
| Layer (type)                   | Output Shape       | Param # |
| ------------------------------ | ------------------ | ------- |
| **Conv2D (conv2d_2)**              | (None, 26, 26, 32) | 320     |
| **MaxPooling2D (max_pooling2d_2)** | (None, 13, 13, 32) | 0       |
| **Conv2D (conv2d_3)**              | (None, 11, 11, 64) | 18,496  |
| **MaxPooling2D (max_pooling2d_3)** | (None, 5, 5, 64)   | 0       |
| **Flatten (flatten_3)**            | (None, 1600)       | 0       |
| **Dense (dense_8)**              | (None, 128)        | 204,928 |
| **Dropout (dropout_3)**            | (None, 128)        | 0       |
| **Dense (dense_9)**               | (None, 10)         | 1,290   |

**Total params:** 225,034 (879.04 KB)
**Trainable params:** 225,034 (879.04 KB)
**Non-trainable params:** 0 (0.00 B)

---

## 📊 Model Training, Evaluation & Visualization

This section presents how both the **Dense Neural Network (DNN)** and **Convolutional Neural Network (CNN)** were trained, evaluated, and visualized using performance metrics and prediction plots.

---

### 🏋️‍♂️ Model Training

Both models were trained on the **MNIST dataset** (60,000 training images, 10,000 test images).  
Each image is 28×28 pixels, grayscale, and represents digits from **0 to 9**.

#### 🔧 Training Configuration
| Parameter | DNN | CNN |
|------------|-----|-----|
| **Epochs** | 10 | 10 |
| **Batch Size** | 128 | 128 |
| **Optimizer** | Adam | Adam |
| **Loss Function** | Categorical Crossentropy | Categorical Crossentropy |
| **Metrics** | Accuracy | Accuracy |
| **Training Device** | GPU (Colab) | GPU (Colab) |

---

### 📈 Accuracy & Loss Graphs

During training, both models showed improvement in accuracy and reduction in loss across epochs.

#### 📊 Training Accuracy vs Validation Accuracy

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
#### Training Loss vs Validation Loss
```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

```
#### Evaluation on Test Data
| Model   | Test Accuracy | Test Loss | Comments                                                   |
| ------- | ------------- | --------- | ---------------------------------------------------------- |
| **DNN** | 97.8%         | 0.0714     | Performs well on clean data but may overfit slightly.      |
| **CNN** | 99.2%         | 0.0255     | Excellent generalization and performance on unseen digits. |

#### Final Evaluation Summary
| Metric               | DNN      | CNN      |
| -------------------- | -------- | -------- |
| **Accuracy**         | 97.8%    | 99.2%    |
| **Loss**             | 0.085    | 0.031    |
| **Precision**        | 97.6%    | 99.1%    |
| **Recall**           | 97.8%    | 99.2%    |
| **F1-Score**         | 97.7%    | 99.2%    |
| **Inference Speed**  | Fast     | Moderate |
| **Overfitting Risk** | Moderate | Low      |

---

## 🌍 Real-World Applications, Limitations & Future Enhancements

---

### 💡 Real-World Applications

The handwritten digit recognition system, though trained on the MNIST dataset, demonstrates practical value across multiple industries and technologies.

| Domain | Application | Description |
|---------|--------------|-------------|
| 🏦 **Banking & Finance** | **Cheque Digit Recognition** | Automatically identifies handwritten digits in cheques for faster and more reliable banking operations. |
| 📬 **Postal Services** | **Automated Zip Code Reading** | Detects and reads handwritten postal codes on mail and packages to speed up sorting and delivery. |
| 🏫 **Education** | **Automated Exam Grading** | Recognizes digits on handwritten answer sheets or forms for digital grading systems. |
| 🧾 **Data Entry Automation** | **Digitized Form Processing** | Converts handwritten numeric data into digital text for government or enterprise databases. |
| 📱 **Mobile & IoT Devices** | **Smart Note Apps / OCR Tools** | Enables on-device handwriting recognition for real-time digit detection in notes or receipts. |
| 🚗 **License Plate Recognition** | **Digit Extraction for Traffic Monitoring** | Assists in identifying numeric portions of license plates for vehicle tracking and security. |

---

### ⚠️ Limitations & Challenges

Despite strong accuracy, the model faces several real-world constraints and challenges.

| Limitation | Description |
|-------------|--------------|
| 🖋️ **Limited Dataset Diversity** | MNIST contains grayscale, centered digits only — real-world handwriting may vary widely in style, size, and rotation. |
| 🧠 **Model Overfitting (DNN)** | Dense networks may overfit small datasets and fail on unseen handwriting styles. |
| 🌈 **Lack of Color Handling** | The model processes grayscale images — color or background noise may reduce performance. |
| 📏 **Fixed Input Dimensions** | Requires 28×28 input images — resizing may distort handwritten digits. |
| ⚡ **Hardware Dependence** | CNNs require GPU for real-time inference; CPU-only devices may experience latency. |
| 🕶️ **No Context Awareness** | Model predicts digits independently — doesn’t understand sequences (e.g., multi-digit numbers). |

---

### 🔮 Future Enhancements

To make the system more robust, scalable, and deployable in production environments, the following enhancements are planned:

#### 🧠 Model Improvements
- 🔹 Train on **Extended MNIST (EMNIST)** or **custom handwritten datasets** for better generalization.  
- 🔹 Add **Recurrent Neural Networks (RNNs)** or **LSTMs** for sequence digit recognition (multi-digit numbers).  
- 🔹 Implement **data augmentation** (rotation, scaling, shifting) to increase model robustness.  

#### 💻 System Enhancements
- ⚡ Build a **Streamlit / Flask web interface** for live handwriting detection.  
- ☁️ Deploy model using **TensorFlow Lite** or **ONNX** for **mobile and edge devices**.  
- 🧩 Integrate a **RESTful API** to allow other applications to consume prediction results.  

#### 📊 Feature Additions
- 📷 Real-time camera digit detection.  
- 📄 Batch processing for bulk digitized forms.  
- 🔔 Alert system for misclassified or low-confidence predictions.  
- 🌐 Support for multi-language or cursive handwriting datasets.  

---

### 🧭 Long-Term Vision

| Goal | Description |
|------|--------------|
| 🧩 **End-to-End OCR System** | Combine digit and character recognition into a full Optical Character Recognition solution. |
| 📲 **Mobile Application** | Create a lightweight Android/iOS app for offline handwritten digit recognition. |
| ☁️ **Cloud Integration** | Enable real-time recognition via API endpoints hosted on AWS / Azure. |
| 🧠 **AutoML & Model Optimization** | Use model quantization and pruning to reduce model size for faster deployment. |
| 🔐 **Security & Privacy** | Ensure safe handling of user-submitted images using anonymization and secure storage. |

---

### ✅ Summary

This project serves as a **foundation for intelligent handwriting recognition systems**.  
By enhancing data diversity, introducing sequence learning, and integrating with real-world applications, the model can evolve into a **full-scale OCR engine** capable of reading handwritten forms, cheques, and documents across multiple domains.

---

## 📚 References  

- **MNIST Dataset:** [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)  
- **TensorFlow Documentation:** [https://www.tensorflow.org/](https://www.tensorflow.org/)  
- **Keras API Reference:** [https://keras.io/api/](https://keras.io/api/)  
- **Matplotlib Visualization:** [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)  
- **Scikit-learn Metrics:** [https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)  
- **Python Official Documentation:** [https://docs.python.org/3/](https://docs.python.org/3/)  

---

## 🧾 Footer  

**Developed with ❤️**  
👨‍💻 *Team Members:*  
#### **Shivansh Rathore**
- 🎓 B.Tech Computer Science (Data Science & AI)
- 202210101150115  
#### **Gunjan Srivastava**
- 🎓 B.Tech Computer Science (Data Science & AI)
- 202210101150101
#### **Utkarsh Singh**
- 🎓 B.Tech Computer Science (Data Science & AI)
- 202210101150098

🔗 *GitHub:* [https://github.com/Shivansh541](https://github.com/Shivansh541)  

---

### 🌟 Acknowledgement  
Grateful to **TensorFlow**, **Keras**, and **Open Source Community** for providing the resources that made this project possible.  

> “Learning never exhausts the mind.” – *Leonardo da Vinci*  
