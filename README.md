# Molecular Activity Prediction 

This project predicts whether a small molecule is biologically active or inactive against SARS-CoV-2, using classical machine learning and deep learning techniques based on cheminformatics features.

## Models Applied

- Random Forest Classifier (scikit-learn)
- Multilayer Perceptron (MLP, PyTorch)

## Data

- Input: Molecule information in a spreadsheet (`Molecules.xlsx`)
- Each row includes:
  - SMILES string
  - Activity label (`Active` or `Inactive`)
  - Set tag (`Train`, `ManualTest`, `Analysis`)
- RDKit is used to compute:
  - Molecular descriptors (e.g., `MolWt`, `LogP`, `TPSA`, `QED`, etc.)
  - Morgan fingerprints (2048-bit circular fingerprints)

## Feature Engineering

- Computed 8 molecular descriptors per molecule.
- Generated 2048-bit Morgan fingerprints.
- Combined all features into a single matrix (`descriptor_cols` + `fp_cols`).

## 1. Random Forest Classifier (scikit-learn)

- Performed train/test split on `Train` set (80/20).
- Trained `RandomForestClassifier` with 100 estimators.
- Evaluated on:
  - Random test set
  - Manual test set
- Predictions made on `Analysis` set with:
  - Predicted label (`Active` / `Inactive`)
  - Confidence score (`predict_proba`)
  - Max Tanimoto similarity to active training molecules

## 2. Multilayer Perceptron (PyTorch)

- Same input features as Random Forest.
- MLP architecture:
  - Input → Linear(256) → ReLU → Linear(128) → ReLU → Linear(1) → Sigmoid
- Training:
  - Loss function: Binary Cross-Entropy (`BCELoss`)
  - Optimizer: Adam
  - Epochs: 100
- Evaluated on:
  - Random test set
  - Manual test set
- Predictions made on `Analysis` set with:
  - Predicted label
  - Confidence score
  - Max Tanimoto similarity

## Metrics and Evaluation

- Accuracy score
- Confusion matrix (visualized with `seaborn`)
- Classification report (precision, recall, F1-score)

## Dependencies

- Python 
- pandas
- numpy
- scikit-learn
- torch
- rdkit
- matplotlib
- seaborn

## Author

Maher Mansur

This project demonstrates how cheminformatics and machine learning can be combined in drug discovery tasks, using simple and interpretable models.

