### Assignment 2 - DVC & MLflow

This assignment demonstrates a reproducible machine learning workflow using:

- **DVC** for data version control  
- **MLflow** for experiment tracking and model versioning  

---

### Part 1 - Data Version Control (prepare.ipynb)

- Loaded raw dataset and saved as `raw_data.csv`
- Split into:
  - `train.csv`
  - `validation.csv`
  - `test.csv`
- Created two data versions using different random seeds:
  - `seed = 99`
  - `seed = 17`
- Tracked all versions using DVC
- Used `git checkout` + `dvc checkout` to restore previous versions
- Printed target class distribution for both data versions
- (Bonus) Configured Google Drive as DVC remote storage

---

### Part 2 -  Experiment Tracking (train.ipynb)

- Created MLflow experiment: `SMS_Spam_Classification`
- Built and registered 3 benchmark models:
  1. Multinomial Naive Bayes  
  2. Logistic Regression  
  3. Linear SVC  
- Logged:
  - Model parameters  
  - Vectorizer parameters  
  - Data seed  
  - AUCPR metric  
- Compared performance across:
  - 3 models  
  - 2 data versions (6 total runs)

---

### Evaluation Metric

Primary metric used: **AUCPR (Area Under Precision-Recall Curve)**  
Suitable for imbalanced spam classification.

---
