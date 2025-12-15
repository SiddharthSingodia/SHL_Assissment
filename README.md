# SHL Intern Hiring Assessment 2025

## Grammar Score Prediction from Speech Audio

## Project Overview

This repository contains a **fully offline baseline solution** for predicting grammar scores from speech audio recordings, developed as part of the **SHL Intern Hiring Assessment 2025**.

The solution extracts **audio-based features** from `.wav` files and trains a **LightGBM regression model** to predict continuous grammar scores. The focus is on **simplicity, stability, and reproducibility**, making it suitable for Kaggle-style evaluation and interviews.

---

## Problem Statement

Given short speech audio recordings, the task is to predict a **continuous grammar score** for each sample.

## Key Constraints

* Offline feature extraction (no pretrained speech models)
* Efficient training and inference
* Robust performance across varied audio quality

---

## Solution Architecture

## 1. Audio Feature Extraction

Audio features are extracted using `librosa` to convert variable-length speech signals into fixed-length numerical representations.

**Extracted Features:**

* MFCCs (Mean and Standard Deviation)
* Spectral Centroid
* Spectral Bandwidth
* Spectral Rolloff
* Zero Crossing Rate

Each audio file is represented as a single feature vector.

---

## 2. Model Selection

**Model Used:** LightGBM Regressor
**Task Type:** Regression

**Why LightGBM?**

* Strong performance on tabular data
* Fast training and inference
* Robust to feature scaling and noise

---

## 3. Model Evaluation

* **Evaluation Metric:** Root Mean Squared Error (RMSE)
* A validation split is used to estimate model performance before final submission

---

## Project Structure

```
├── notebook.ipynb          # End-to-end training & inference notebook
├── train.csv               # Training metadata (audio paths + labels)
├── test.csv                # Test metadata
├── audio/                  # Audio files
├── submission.csv          # Model predictions
└── README.md               # Project documentation
```

---

## Setup Instructions

## Python Version

* Python **3.9 or above** recommended

## Dependencies

```bash
pip install numpy pandas librosa lightgbm scikit-learn tqdm
```

---

## How to Run the Project

1. Ensure all audio file paths in `train.csv` and `test.csv` are correct
2. Open `notebook.ipynb`
3. Run all cells sequentially:

   * Load dataset
   * Extract audio features
   * Train LightGBM model
   * Validate performance
   * Generate submission file
4. The output will be saved as `submission.csv`

---

## Results and Performance

* **Validation RMSE:** ~0.75 (may vary depending on data split)
* Provides a stable and reproducible baseline

---

## Limitations

* Does not use linguistic or semantic information
* Relies only on handcrafted audio features
* Performance may degrade on highly noisy audio

---

## Future Improvements

* Add prosodic features (pitch, energy, speaking rate)
* Include delta and delta-delta MFCCs
* Try ensemble models (CatBoost, XGBoost)
* Explore deep learning models using log-mel spectrograms

---

## Notes

* This is a **fully offline solution**
* Designed as a **strong baseline**, not a final optimized model

---

## Author

**Siddharth Singodia**
Final-year B.Tech Student | Machine Learning & Audio Processing
