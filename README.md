# Audio Anomaly Detection using Machine Learning

This project applies supervised machine learning models to detect anomalies in industrial fan sounds using audio features extracted from the [MIMII Dataset](https://zenodo.org/record/3384388).

## ?? Project Overview

- **Dataset:** MIMII (fan 0dB subset)
- **Task:** Binary classification (normal vs. abnormal)
- **Features:** Extracted MFCCs, chroma, and spectral features
- **Models Tested:** Logistic Regression, Random Forest, Isolation Forest
- **Balancing:** SMOTE used to address class imbalance
- **Best Model:** Random Forest + SMOTE (F1 Score: 0.842, ROC AUC: 0.947 on test set)

## ?? Performance Metrics

| Model                     | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------------|----------|-----------|--------|----------|---------|
| Logistic (Unbalanced)    | 0.873    | 0.897     | 0.634  | 0.743    | 0.867   |
| Random Forest (Unbalanced)| 0.891   | 0.947     | 0.659  | 0.777    | 0.954   |
| Logistic (Balanced)      | 0.803    | 0.627     | 0.780  | 0.696    | 0.868   |
| Random Forest (Balanced) | 0.898    | 0.982     | 0.659  | 0.788    | 0.948   |
| Logistic (SMOTE)         | 0.796    | 0.618     | 0.768  | 0.685    | 0.866   |
| Random Forest (SMOTE)    | **0.915**| **0.914** | **0.780**| **0.842**| **0.947**|
| Isolation Forest         | 0.694    | 0.468     | 0.439  | 0.453    | 0.676   |

## ?? Repository Contents

- `anomaly_detection_audio.ipynb` – Jupyter notebook with full analysis and modeling
- `README.md` – Project overview and instructions
- `.gitignore` – File exclusions for Git
- `requirements.txt` – Python package dependencies (optional)

## ?? Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME