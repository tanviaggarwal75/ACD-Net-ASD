# ACD-Net: Adaptive Channel-Aware Deep Learning for fMRI Connectome Analysis in Autism Spectrum Disorder Classification

## Overview

This repository contains the implementation of **ACD-Net (Adaptive Channel-Aware Deep Learning Network)**, a deep learning framework for Autism Spectrum Disorder (ASD) classification using resting-state functional MRI (rs-fMRI) functional connectivity matrices.

The proposed architecture combines Convolutional Neural Networks (CNNs) with a customized **Channel Attention Mechanism** to adaptively emphasize informative connectivity representations and improve ASD classification performance.

The repository also includes a **CNN + Multi-Head Attention (CNN-MHA)** baseline model used for comparative evaluation.

---

## Dataset

This study utilizes the **Autism Brain Imaging Data Exchange (ABIDE)** dataset.

### Data Sources

ABIDE:
https://fcon_1000.projects.nitrc.org/indi/abide/

ABIDE Preprocessed Connectomes Project (PCP):
http://preprocessed-connectomes-project.org/abide/

### Brain Atlas

* Craddock 200 (CC200) Atlas
* 200 Regions of Interest (ROIs)

### Input Data

The models expect precomputed functional connectivity matrices of shape:

(samples, 200, 200)

which are reshaped into:

(samples, 200, 200, 1)

for CNN processing.

---

## Repository Structure

```text
ACD-Net-ASD/
│
├── README.md
├── requirements.txt
│
├── acd_net.py
│
├── baselines/
│   └── cnn_mha.py
│
└── figures/
    ├── roc_curve.png
    └── attention_weights.png
```

---

## Models Included

### 1. ACD-Net (Proposed Model)

Architecture:

* Conv2D (32)
* MaxPooling
* Conv2D (64)
* MaxPooling
* Conv2D (128)
* MaxPooling
* Channel Attention Module
* Dense (128)
* Dropout (0.5)
* Sigmoid Output Layer

Evaluation Metrics:

* Accuracy
* AUC
* F1 Score

The implementation includes extraction and visualization of channel-attention weights to improve model interpretability.

---

### 2. CNN + Multi-Head Attention (Baseline)

Architecture:

* CNN Feature Extractor
* Multi-Head Attention Layer
* Dense Layers
* Sigmoid Output Layer

Hyperparameter optimization is performed using Bayesian Optimization.

Evaluation Metrics:

* Accuracy
* AUC
* F1 Score

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Requirements

Main libraries used:

```text
tensorflow
keras-tuner
numpy
scikit-learn
matplotlib
seaborn
```

---

## Usage

### Run ACD-Net

```bash
python acd_net.py
```

### Run CNN + Multi-Head Attention Baseline

```bash
python baselines/cnn_mha.py
```

---

## Figures

The `figures/` directory contains representative visualizations from the study:

* **ROC Curve** illustrating the classification performance of the proposed ACD-Net model.
* **Channel Attention Weight Visualization** demonstrating the feature importance learned by the attention mechanism.

These figures are included to support the experimental findings and interpretability analysis presented in the accompanying manuscript.

---

## Notes

This repository does not include dataset download or preprocessing scripts.

Users should independently obtain the ABIDE PCP dataset and generate functional connectivity matrices prior to model training and evaluation.

---

## Authors

**Tanvi Aggarwal**
Department of Artificial Intelligence and Data Sciences
Indira Gandhi Delhi Technical University for Women (IGDTUW)

**Dr. Ritika Kumari**
Department of Artificial Intelligence and Data Sciences
Indira Gandhi Delhi Technical University for Women (IGDTUW)
