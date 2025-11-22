# Image Segmentation Research Project

Image segmentation research project implementing **Mask R-CNN**, **PSPNet**, and an **MST-based spectral matting** approach.

This repository accompanies the research work documented in `Image Segmentation Research Paper.pdf` and provides reference implementations and experiments for comparing these three segmentation methods.

---

## Overview

This project explores and compares three image segmentation approaches:

- **Mask R-CNN** – instance segmentation using region proposals and per-instance masks.
- **PSPNet (Pyramid Scene Parsing Network)** – semantic segmentation using pyramid pooling to capture multi-scale context.
- **MST-based Spectral Matting** – a spectral segmentation / matting approach based on a Minimum Spanning Tree (MST) formulation.

The goal is to provide a clean, reproducible codebase for running experiments, reproducing results, and extending the work to new datasets or model variants.

---

## Repository Structure

```text
Image-Segmentation-Research-Project/
├── maskrcnn/                              # Mask R-CNN implementation & scripts
├── pspnet/                                # PSPNet implementation & scripts
├── spectral_matting/                      # MST-based spectral matting implementation
├── requirements.txt                       # Python dependencies
└── Image Segmentation Research Paper.pdf  # Project report/paper
