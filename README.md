# Rationale Extraction with Knowledge Distillation (REKD)

This repository contains the official code implementation for **Learn from A Rationalist: Distilling Intermediate Interpretable Rationales**. 

*(Note: Reference and specific paper details will be added when the paper gets published online.)*

## 📑 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Ablation Studies](#ablation-studies)
- [Citation](#citation)
- [License](#license)

## 📖 Overview
*In the select-predict architecture of rationale extraction (RE), the generator relies on the guidance of the predictor to select important features (i.e., a rationale) while the predictor relies on the output of the generator to learn task prediction. This "chicken-and-egg" dilemma is significantly exacerbated when the base neural networks are not sufficiently capable. To mitigate this, we propose a knowledge distillation method REKD for Gumbel-Softmax based RE models where a student models learns from the rationales and the predictions of a teacher RE model in addition to its own RE exploration. Our approach provides a neural-model agnostic distillation framework that leverages the intrinsic curriculum of the Gumbel-Softmax annealing. We validate REKD on both language and vision tasks using multiple variants of BERT and ViT as RE backbones. Experiments demonstrate that REKD significantly improves the predictive performance of the student RE models.*

<figure align="center">
  <img src="repo_assets/REKD_schematic.svg" alt="The schematic of REKD" width="800">
  <figcaption><b>Figure:</b> Architecture Schematic of REKD. </figcaption>
</figure>

## 🗂 Repository Structure
The repository is organized as follows:

```text
REKD/
├── ablation_scripts/   # Bash scripts to run ablation studies
├── data/               # Directory containing datasets and data processing scripts
├── nns/                # Neural network modules and model architectures
├── results/            # Output directory for saving evaluation results
├── run/                # Execution scripts for training and testing
├── saved/              # Directory for saving model checkpoints
├── scripts/            # Bash scripts for running main experiments
├── utils/              # Helper functions, metrics, and utility scripts
├── requirements.txt    # Python dependencies required to run the project
└── README.md           # This documentation file
```

## ⚙️ Installation
Follow these steps to set up the environment and install the required dependencies for REKD.

**Prerequisite:** Python 3.11.5; cuda 12.6

**1. Clone the repository and install dependencies**

```bash
git clone https://github.com/JiayiDai/REKD.git
cd REKD
pip install -r requirements.txt
```

## 📊 Data Preparation
The REKD framework is validated on both language and vision tasks. Please prepare the datasets before running the experiments.

See data folder for details.
