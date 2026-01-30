YOLOv10-WT: A Lightweight Deep Learning Framework for Real-time Smoldering Forest Fire Detection on Edge Devices


Overview
YOLOv10-WT is a lightweight deep learning framework specifically designed for detecting smoldering forest fires in thermal infrared images acquired by unmanned aerial vehicles (UAVs). The framework achieves a balance between high detection accuracy and lightweight deployment by synergistically integrating wavelet transform-based feature enhancement, lightweight convolution modules, and attention mechanisms.


Problem Statement
Smoldering forest fires are characterized by:
Low combustion temperature
Weak thermal radiation signatures
Prolonged latency periods
Difficult early detection before escalation to large-scale wildfires


This work proposes a UAV-based solution using thermal infrared imaging with lightweight deep learning.
The framework is improved based on yolov10, which can be downloaded and reproduced by yourself. The core modules are located in custom_modules_ds.py, which can be further improved.
Architecture
The GitHub of Yolo ontology is: https://github.com/ultralytics/ultralytics
Core Components
Wavelet-based Feature Enhancement Module (C2f_WT)

Lightweight Convolution (GhostConv)

Dual-branch ECA Attention Mechanism

GatedSConv Spatial Gating

The third-party library used in this project is in requirements.txt

Usage
Training：
python train.py

Dataset

5,128 UAV-collected long-wave thermal infrared images
The data used in this project is the smoldering fire data set taken by ourselves. If you have any questions, please contact us.
