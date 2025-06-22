# Molecular Activity Prediction Using Random Forest (SARS-CoV-2 Case Study)

This project predicts whether a small molecule is biologically active or inactive against SARS-CoV-2, using classical machine learning techniques based on cheminformatics features.

## 🧪 Overview

- **Type**: Binary classification (`Active` vs `Inactive`)
- **Technique**: Classical ML using `RandomForestClassifier` from scikit-learn
- **Features Used**:
  - 8 RDKit descriptors (e.g. MolWt, TPSA, LogP, QED)
  - 2048-bit Morgan fingerprints

## 📂 Dataset Structure

Molecules are divided into 4 sets:
- **Training**: Used to train the model
- **Random Test **: Selected by random split from training data
- **Manual Test **: Manually selected for independent validation
- **Analysis **: Used to demonstrate predictions on unseen data

## ⚙️ Workflow

1. **Data Preprocessing**:
   - Removed duplicates and invalid SMILES
   - Computed descriptors + Morgan fingerprints

2. **Model Training**:
   - Trained a Random Forest on training data
   - Evaluated using confusion matrix and accuracy on the train and test sets

3. **Manual and Analysis Set Evaluation**:
   - Predicted activity and confidence scores
   - Computed Tanimoto similarity to active compounds

## Outputs

- Confusion matrix (visualized with heatmap)
- Prediction accuracy
- Analysis output includes:
  - Predicted activity
  - Confidence (probability)
  - Tanimoto similarity to training actives

## Tools & Libraries

- Python, Pandas, NumPy
- RDKit (for chemical processing)
- scikit-learn (for ML)
- Matplotlib, Seaborn (for visualization)

## Future Work

- Rebuild the project in **PyTorch** using a neural network (MLP)
- Explore deep learning on SMILES (e.g., RNN or GNN models)
- Add more molecules for larger-scale evaluation

---

### 💡 Author Notes

This project demonstrates how cheminformatics and machine learning can be combined in drug discovery tasks, using simple and interpretable models.

