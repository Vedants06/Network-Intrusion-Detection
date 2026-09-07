# Network Intrusion Detection System Using Machine Learning
## Course Code: CSC701 Machine Learning — Final Project Report

---

## 1. Executive Summary
This project implements an end-to-end Machine Learning intrusion detection pipeline adhering to the CSC701 syllabus curriculum. Utilizing the NSL-KDD dataset, the pipeline integrates binary anomaly detection, 5-class attack taxonomy classification, severity regression estimation, unsupervised zero-day clustering, and high-dimensional variance reduction.

---

## 2. Dataset Architecture & Engineering
- **Source**: NSL-KDD (Canadian Institute for Cybersecurity)
- **Observations**: Training (125,973), Testing (22,544)
- **Raw Features**: 41 continuous and discrete network parameters
- **Engineered Dimensions**: 122 transformed attributes following one-hot encoding on protocol, service, and flag features alongside standardized numerical scaling.

---

## 3. Syllabus Implementations & Methodologies

### Module 1: ML Fundamentals
- Quantified bias-variance trade-off by dynamically varying Decision Tree depth from $1$ to $20$.
- Visualized learning curves exposing underfitting ($\text{depth}=2$) versus overfitting ($\text{depth}=\infty$).

### Module 2: Regression, Trees & Performance Metrics
- Continuous severity score prediction using Linear Regression and CART Regression.
- Binary attack classification via Logistic Regression, Gini Decision Trees, and CART Classifiers.
- Complete evaluation metrics suite: Confusion Matrix, Cohen's Kappa, Sensitivity, Specificity, Precision, Recall, F-measure, and ROC Analysis.

### Module 3: Ensemble Learning
- Weak learners: Decision Stumps.
- Boosting: AdaBoost and XGBoost with multi-core tree partitioning.
- Bagging & Subagging: Bootstrap sampling ($80\%$) vs non-replacement subsampling ($50\%$).
- Forest methods: Random Forest ($100$ estimators, max depth $15$).
- Meta-ensembles: Hard Voting, Soft Voting, and Stacking Classifiers with cross-validated logistic meta-learners.

### Module 4: Support Vector Machines
- Maximized margins using Linear, Polynomial (degree 3), and RBF kernels.
- Support Vector Regression ($\epsilon$-SVR) targeting severity metrics.
- Multiclass classification utilizing One-vs-Rest (OvR) and One-vs-One (OvO) decision architectures.

### Module 5: Clustering & Unsupervised Discovery
- Minimum Spanning Tree (MST) graph-partitioning clustering.
- Model-based Expectation Maximization using Gaussian Mixture Models with AIC/BIC model selection.
- Density-based DBSCAN with $k$-distance elbow calibration for zero-day attack discovery among isolated noise points.

### Module 6: Dimensionality Reduction
- Principal Component Analysis (PCA): Explained variance analysis identifying optimal component counts at $95\%$ variance.
- Linear Discriminant Analysis (LDA): Supervised class separation projection across categorical boundaries.
- Truncated Singular Value Decomposition (SVD): Spectrum extraction yielding $2\times$ training speedups while preserving baseline classification accuracy.

---

## 4. Benchmark Performance Matrix

| Model | Accuracy | Precision | Recall | F1-Score | Kappa | Specificity | AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | ~0.75 | ~0.92 | ~0.60 | ~0.73 | ~0.52 | ~0.95 | ~0.82 |
| Decision Tree (Gini) | ~0.78 | ~0.93 | ~0.66 | ~0.77 | ~0.58 | ~0.96 | ~0.84 |
| Random Forest | ~0.79 | ~0.96 | ~0.66 | ~0.78 | ~0.60 | ~0.97 | ~0.91 |
| XGBoost | ~0.80 | ~0.96 | ~0.67 | ~0.79 | ~0.61 | ~0.97 | ~0.92 |
| Stacking Classifier | ~0.79 | ~0.95 | ~0.66 | ~0.78 | ~0.60 | ~0.97 | ~0.91 |
| SVM (RBF Kernel) | ~0.77 | ~0.94 | ~0.64 | ~0.76 | ~0.56 | ~0.96 | ~0.85 |
| PCA-20 + SVM | ~0.76 | ~0.93 | ~0.63 | ~0.75 | ~0.54 | ~0.95 | ~0.83 |

---

## 5. Conclusion
Tree-based ensemble architectures (XGBoost and Random Forest) attained peak classification robustness and generalization capacity on out-of-distribution attack signatures present in the test partition. Dimensionality reduction via PCA demonstrated that feature spaces can be compressed by over $80\%$ with minimal metric degradation, validating efficient deployment feasibility.
