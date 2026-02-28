#  Low-Latency Audio Machine Learning Pipeline

A production-style **audio event classification pipeline** designed for **real-time, low-latency inference**, inspired by on-device audio use cases such as **headphones and smart audio systems**.

This project covers the **entire applied ML lifecycle** — from dataset ingestion and model training to ONNX export, latency benchmarking, and API-based inference.

---

##  Key Highlights

- 🎵 **Audio Event Classification** using MFCC features + lightweight CNN  
- ⚡ **Ultra-low latency inference**: **0.039 ms (batch=1)** via ONNX Runtime  
- 📊 **~94% test accuracy** on Speech Commands v0.02 dataset  
- 🍎 **Apple MPS backend** used for local training on macOS  
- 🔁 End-to-end pipeline: training → evaluation → export → deployment  

---

## Problem Statement

Real-time audio systems (e.g., headphones, wearables, embedded devices) require:
- Extremely **low inference latency**
- Small, efficient models
- Reliable performance under tight compute constraints

This project demonstrates how to design an **ML pipeline optimized for such constraints** while maintaining strong accuracy.

---

##  Project Architecture


audio-ml-pipeline/
-├── src/ # Training, evaluation, ONNX export, benchmarking
-├── api/ # FastAPI inference service
-├── models/ # Trained model, ONNX export, confusion matrix
-├── data/ # Dataset (excluded from repo)
-├── requirements.txt
-├── download_data.py
-└── README.md

---

##  Model & Performance

- **Model**: MFCC feature extractor + lightweight CNN
- **Classes**: `speech`, `noise`, `silence`
- **Test Accuracy**: ~93.8%
- **ONNX Inference Latency**: **0.039 ms (batch=1)**

Confusion matrix is available in `models/confusion_matrix.png`.

---



