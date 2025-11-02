# 🐦 Bird Sound Detector

This project is a simple **Machine Learning application** that detects and classifies **bird sounds**.  
It uses two deep learning models — **ResNet50** and **MobileNetV2** — to train on bird sound datasets and predict the bird species when a new audio clip is given as input.

---

## 📘 Overview
The main goal of this project is to identify bird species based on their recorded sounds.  
Audio files are processed using **MFCC feature extraction** and passed into pre-trained CNN architectures (ResNet50 and MobileNetV2) for classification.

---

## ⚙️ How It Works
1. **Dataset Preparation** – Bird sound recordings are collected from online datasets.  
2. **Feature Extraction** – Each audio file is converted into numerical features using MFCCs.  
3. **Model Training** – The extracted features are trained using both **ResNet50** and **MobileNetV2** models.  
4. **Prediction** – When a new bird sound is uploaded, the trained model predicts the corresponding bird species.

---

## 🧠 Models Used
- **ResNet50** – Deep CNN architecture effective for high-accuracy classification.  
- **MobileNetV2** – Lightweight and fast model suitable for mobile or real-time predictions.

---

## 🖥️ Files in This Project
- `train.py` – Trains both ResNet50 and MobileNetV2 models on the dataset.  
- `app.py` – Simple web interface to upload and predict bird sounds.  
- `model_resnet50.h5` & `model_mobilenetv2.h5` – Trained model files.
- Dataset used : https://www.kaggle.com/datasets/soumendraprasad/sound-of-114-species-of-birds-till-2022

---

## 🚀 Future Improvements
- Add more bird species and larger datasets.  
- Deploy the app on a cloud platform.  
- Add spectrogram visualization for uploaded sounds.

---

## 📄 License
This project is open-source and free to use for educational or research purposes.
