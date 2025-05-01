# Engagement Prediction

This repository contains the code, outputs, project report, survey (DA-1 and survey paper draft), and presentation for our engagement prediction project.

This project uses computer vision and deep learning techniques to predict the emotional state and engagement level of individuals in a video. It processes frames from video input and performs emotion classification using pre-trained models.

---

## Team

- **Ayushi Jha** - 22BCE1980  
- **Soham Amberkar** - 22BCE1770

---

## 🚀 Features

- Detect faces in video frames
- Predict engagement level (e.g., attentive vs. distracted)
- Recognize basic emotions like *Happy*, *Sad*, *Angry*, etc.
- Visualization of predictions over time

---

## 🛠️ Major Frameworks

- **PyTorch** – For everything related to deep learning
- **OpenCV**
- **Deep Learning (CNNs)**
- **FER (Facial Emotion Recognition) library**
- **Numpy** – For array manipulation and algebra
- **Matplotlib** – For visualization

---

## 📓 Notebook Preview

Try it out in Colab:  
[engagement_prediction.ipynb](https://colab.research.google.com/github/ayushi-2104/engagement-prediction/blob/main/engagement_prediction.ipynb)

---

## 📂 Dataset Used

- **DAiSEE:** Dataset for Affective States in E-Environments. Contains ~9000 videos annotated with four engagement levels.
- **EmotiW Student Engagement Subchallenge Dataset:** Real-world data of students in online learning environments labeled with engagement levels from 0 to 1.

---

## 📈 Model & Method

- **Self-supervised Facial Masked Autoencoder (FMAE) architecture**
- Custom facial masking strategy focusing on eyes, nose, and mouth
- Reconstruction and adversarial loss for learning fine-grained features
- Fine-tuning on engagement classification using pretrained features

---

## 🧪 Results

### DAiSEE Dataset

| Model                | Accuracy   |
|----------------------|------------|
| ResNet + TCN         | 83.24%     |
| Optimized ShuffleNet v2 | 84.12%  |
| **FMAE (Ours)**      | **84.96%** ✅ |

### EmotiW Dataset

| Model           | MSE      |
|-----------------|----------|
| MAGRU (Best baseline) | 0.0517 |
| **FMAE (Ours)** | **0.0629** ✅ |

> Our method achieves state-of-the-art or competitive performance without requiring labeled data during pretraining.

---

## 🎯 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Mean Squared Error (MSE) for regression-based engagement scores

---

## 🏃‍♂️ Steps to Run

### In Cloud/Kaggle/Colab (Recommended)

- Run the notebook as-is in Kaggle (with GPU enabled) or Colab; most prerequisites are pre-installed.
- Other required packages are installed in the first cell of the notebook.
- All cells will execute without error, but the model will be retrained with randomly initialized weights each time you run the notebook. So, results may slightly vary from the reported ones.

### Local (Linux Required)

- Install PyTorch with CUDA enabled, numpy, pillow, matplotlib, and all other packages mentioned in the first cell.
- Ensure sufficient resources: **VRAM (8 GB)**, **RAM (8 GB)**, and at least **1 GB of free disk space**.

---

