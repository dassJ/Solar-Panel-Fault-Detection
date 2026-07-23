# Solar Panel Fault Detection and Analysis Using Deep Learning

## Project Overview

This project presents an AI-based system for detecting faults in solar panels using thermal infrared images. A deep learning model is trained to classify solar panels into four categories: Healthy, Cracked, Hot-Spot, and Dust-Covered. The project also includes Grad-CAM visualization for explainability, fault severity estimation, and a Streamlit web application for real-time prediction.

## Problem Statement

Manual inspection of large-scale solar farms is time-consuming, expensive, and prone to human error. Traditional inspection methods cannot efficiently detect faults across thousands of solar panels. An automated solution is required to improve inspection speed and maintenance efficiency.

## Solution

The proposed system uses transfer learning with a ResNet-18 model to classify thermal images of solar panels. It generates confidence scores, Grad-CAM heatmaps, and maintenance alerts through a browser-based Streamlit interface.

## Technologies Used

- Python
- PyTorch
- TorchVision
- OpenCV
- Streamlit
- Matplotlib
- Scikit-learn
- ResNet-18
- MobileNetV3

## Features

- Solar panel fault classification
- Thermal image analysis
- Grad-CAM visualization
- Fault severity estimation
- Confidence score prediction
- Maintenance alert generation
- Streamlit web interface
- Lightweight MobileNetV3 comparison

## Fault Classes

- Healthy
- Cracked
- Hot-Spot
- Dust-Covered

## Project Report

The complete project documentation, implementation details, algorithms, and experimental results are available below.

📄 **[Solar Project Report](Solar_Project_Report.pdf)**

## Conclusion

This project demonstrates an end-to-end deep learning solution for automated solar panel fault detection using thermal images. The system combines accurate fault classification, explainable AI, and an interactive web interface to support efficient solar panel inspection and maintenance.
