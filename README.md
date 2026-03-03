## Applied Machine Learning - Jan-Apr 2026

Repository for coursework assignments in **Applied Machine Learning**.

### Structure

### Project Structure

```text
Applied-Machine-Learning/
│
├── Assignment_1/
│   ├── prepare.ipynb            # Data loading, exploration, preprocessing
│   ├── train.ipynb              # Model training and evaluation
│   ├── SMSSpamCollection        # Original dataset
│   ├── train.csv                # Training split
│   ├── validation.csv           # Validation split
│   └── test.csv                 # Test split
│
├── Assignment_2/
│   ├── prepare.ipynb            # Data versioning using DVC
│   ├── train.ipynb              # Experiment tracking using MLflow
│   └── dvc.yaml / dvc.lock      # (if applicable)
│
└── Assignment_3/
    ├── app.py                   # Flask application for model serving
    ├── score.py                 # Scoring function with threshold logic
    ├── test.py                  # Unit and integration tests (pytest)
    ├── CSVC_best_model.pkl      # Final calibrated LinearSVC model
    └── coverage.txt             # Test coverage report
     
