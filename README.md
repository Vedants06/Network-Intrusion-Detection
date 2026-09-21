# Network Intrusion Detection System Using Machine Learning (CSC701)

An end-to-end Machine Learning pipeline for network traffic anomaly detection, attack category classification, severity estimation, and zero-day threat discovery on the NSL-KDD dataset.

---

## 📁 Repository Structure

```
Network-Intrusion-Detection-ML/
├── data/
│   ├── raw/                  # KDDTrain+.txt and KDDTest+.txt
│   └── processed/            # Processed arrays and csv files
├── notebooks/                # Numbered notebooks 01 through 10
├── src/                      # Modular Python processing and model modules
├── models/
│   ├── saved_models/         # Serialized .pkl models
│   └── encoders/             # Preprocessors and label encoders
├── plots/                    # Output visual plots organized by module
├── results/                  # Metric summaries and benchmark CSVs
├── docs/                     # Detailed project report
├── requirements.txt
└── setup_project.py
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python setup_project.py
pip install -r requirements.txt
```

### 2. Data Acquisition
Download `KDDTrain+.txt` and `KDDTest+.txt` from [Kaggle NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) and place them into `data/raw/`.

### 3. Pipeline Execution
Execute notebooks sequentially:
1. `notebooks/01_Data_Preprocessing.ipynb`
2. `notebooks/02_EDA_and_Visualization.ipynb`
3. `notebooks/03_Module1_ML_Fundamentals.ipynb`
4. `notebooks/04_Module2_Regression_and_Trees.ipynb`
5. `notebooks/05_Module3_Ensemble_Learning.ipynb`
6. `notebooks/06_Module4_SVM_Classification.ipynb`
7. `notebooks/07_Module5_Clustering.ipynb`
8. `notebooks/08_Module6_Dimensionality_Reduction.ipynb`
9. `notebooks/09_Final_Model_Comparison.ipynb`
10. `notebooks/10_Complete_Pipeline.ipynb`
