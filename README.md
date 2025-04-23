# engagement-prediction
This repository contains the code, output of our project as well as the report, survey (DA-1 and survey paper draft), presentation. Read below for more info

This project uses computer vision and deep learning techniques to predict the emotional state and engagement level of individuals in a video. It processes frames from video input and performs emotion classification using pre-trained models.

Team
Ayushi Jha-22BCE1980
Soham Amberkar-22BCE1770

🚀 Features
Detect faces in video frames
Predict engagement level (e.g., attentive vs. distracted)
Recognize basic emotions like Happy, Sad, Angry, etc.
Visualization of predictions over time

Major frameworks
PyTorch - For everything related to deep learning
OpenCV
Deep Learning (CNNs)
FER (Facial Emotion Recognition) library
Numpy - For some array manipulation and algebra
Matplotlib - For visualization

📓 Notebook Preview
You can try it out in Colab:
https://colab.research.google.com/github/SohamAmberkar/dip/blob/main/docs/tutorials/python/Predict%20engagement%20and%20emotions%20on%20video.ipynb

Steps to run
In Cloud/Kaggle/Colab (recommended)

You can just run the notebook as is, in Kaggle (with GPU enabled) or Colab as they'd have most prerequisites installed. Other packages that maybe required are installed in the first cell of the notebook
All the cells will execute without any error, though the model will be retrained with randomly initialized weights each time you run the notebook. So the results reported and the results obtained might slightly vary
Local (Linux required for the code to automatically download the dataset)

Install PyTorch with cuda enabled, numpy, pillow, matplotlib alongside everything in the first cell
Make sure there is sufficient VRAM (8 GB) and RAM (8 GB) with atleast 1 GB of free disk space
