# Image Segmentation Research Project

Image segmentation research project implementing **Mask R-CNN**, **PSPNet**, and an **MST-based spectral matting** approach.

This repository accompanies the research work documented in Image Segmentation Research Paper.pdf and provides reference implementations and experiments for comparing these three segmentation methods.

---

## Overview

This project explores and compares three image segmentation approaches:

- **Mask R-CNN** – instance segmentation using region proposals and per-instance masks.
- **PSPNet (Pyramid Scene Parsing Network)** – semantic segmentation using pyramid pooling to capture multi-scale context.
- **MST-based Spectral Matting** – a spectral segmentation / matting approach based on a Minimum Spanning Tree (MST) formulation.

The goal is to provide a clean, reproducible codebase for running experiments, reproducing results, and extending the work to new datasets or model variants.

---

## Datasets

The following datasets were used for training, evaluation, and benchmarking in this research project:

### **1. PASCAL VOC 2012**
- Used primarily for **PSPNet** training and evaluation.
- Includes 20 object classes plus background.
- Serves as a standard benchmark for semantic segmentation tasks.

### **2. COCO 2017 (Common Objects in Context)**
- Used for **Mask R-CNN** because it provides instance-level annotations.
- Contains 80 object categories with bounding boxes and segmentation masks.
- Well-suited for instance segmentation research and comparison.

### **3. BSDS500 (Berkeley Segmentation Dataset)**
- Used to evaluate the **MST-based Spectral Matting** approach.
- Provides high-quality contour and boundary annotations.
- Commonly used for matting, boundary detection, and perceptual segmentation studies.

---

## Repository Structure

```text
Image-Segmentation-Research-Project/
├── maskrcnn/                              # Mask R-CNN implementation & scripts
├── pspnet/                                # PSPNet implementation & scripts
├── spectral_matting/                      # MST-based spectral matting implementation
├── requirements.txt                       # Python dependencies
└── Image Segmentation Research Paper.pdf  # Project report/paper
