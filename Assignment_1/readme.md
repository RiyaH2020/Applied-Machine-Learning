## SMS Spam Classification Prototype (Assignment 1)

This repository contains a prototype for SMS spam classification.

### Notebooks

### `prepare.ipynb`

Functions to:

- Load the data from a given file path
- Preprocess the data (if needed)
- Split the data into train/validation/test sets
- Store the splits at:
  - `train.csv`
  - `validation.csv`
  - `test.csv`

### `train.ipynb`

Functions to:

- Fit a model on train data
- Score a model on given data
- Evaluate the model predictions
- Validate the model
- Fit on train data
- Score on train and validation data
- Evaluate on train and validation data
- Fine-tune hyperparameters using train and validation data (if necessary)
- Score three benchmark models on test data and select the best one
