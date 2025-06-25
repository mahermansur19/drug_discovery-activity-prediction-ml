import pandas as pd

# Load molecule data
dtfm = pd.read_excel("Molecules.xlsx")

# Drop missing values and duplicates
dtfm = dtfm.dropna()
dtfm_clean = dtfm.drop_duplicates()

# Check SMILES validity
from rdkit import Chem
dtfm_clean["Mol"] = dtfm_clean["SMILES"].apply(Chem.MolFromSmiles)
dtfm_clean = dtfm_clean[dtfm_clean["Mol"].notnull()].reset_index(drop=True)

# Compute descriptors
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
descriptor_funcs = [
    ("MolWt", Descriptors.MolWt),
    ("LogP", Crippen.MolLogP),
    ("NumHAcceptors", Lipinski.NumHAcceptors),
    ("NumHDonors", Lipinski.NumHDonors),
    ("TPSA", Descriptors.TPSA),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("QED", QED.qed)
]
for name, func in descriptor_funcs:
    dtfm_clean[name] = dtfm_clean["Mol"].apply(func)

# Convert activity labels to binary
dtfm_clean["Activity_values"] = dtfm_clean["Activity"].replace({"Active": 1, "Inactive": 0})

# Compute Morgan fingerprints
from rdkit.Chem import AllChem
def mol_to_fp(mol, radius=2, nBits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)

fps = dtfm_clean["Mol"].apply(mol_to_fp)
dtfm_clean["FP_object"] = fps

# Convert fingerprint objects to numpy arrays
import numpy as np
fp_array = np.array([np.array(fp) for fp in fps])
fp_dtfm = pd.DataFrame(fp_array, columns=[f'FP_{i}' for i in range(fp_array.shape[1])])

# Combine all features
dtfm_final = pd.concat([dtfm_clean.reset_index(drop=True), fp_dtfm], axis=1)
descriptor_cols = ['MolWt', 'LogP', 'NumHAcceptors', 'NumHDonors', 'TPSA', 'NumRotatableBonds', 'RingCount', 'QED']
fp_cols = [f'FP_{i}' for i in range(2048)]
feature_cols = descriptor_cols + fp_cols

# Train-test split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_dtfm = dtfm_final[dtfm_final["Set"] == "Train"]
x = train_dtfm[feature_cols]
y = train_dtfm["Activity_values"]
x_train, x_randtest, y_train, y_randtest = train_test_split(x, y, test_size=0.2, random_state=188)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=19)
rf_model.fit(x_train, y_train)

# Evaluate model
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, title):
    labels = ['Inactive', 'Active']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Training set evaluation
train_preds = rf_model.predict(x_train)
cfmx_train = confusion_matrix(y_train, train_preds)
print("Training Accuracy:", accuracy_score(y_train, train_preds))
print("\nConfusion Matrix (Train):\n", cfmx_train)
plot_confusion_matrix(cfmx_train, "Confusion Matrix (Train Set)")

# Random test set evaluation
randtest_preds = rf_model.predict(x_randtest)
cfmx_randtest = confusion_matrix(y_randtest, randtest_preds)
print("Random Test Accuracy:", accuracy_score(y_randtest, randtest_preds))
print("\nConfusion Matrix (Test):\n", cfmx_randtest)
plot_confusion_matrix(cfmx_randtest, "Confusion Matrix (Random Test Set)")

# Manual test set evaluation
manual_dtfm = dtfm_final[dtfm_final["Set"] == "ManualTest"]
x_manual = manual_dtfm[feature_cols]
y_manual = manual_dtfm["Activity_values"]
manual_preds = rf_model.predict(x_manual)
print("Manual Test Accuracy:", accuracy_score(y_manual, manual_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_manual, manual_preds))
print("\nClassification Report:\n", classification_report(y_manual, manual_preds))

# Analysis predictions
analysis_dtfm = dtfm_final[dtfm_final["Set"] == "Analysis"]
x_analysis = analysis_dtfm[feature_cols]
analysis_preds = rf_model.predict(x_analysis)
analysis_probs = rf_model.predict_proba(x_analysis)

analysis_dtfm["Predicted_Activity"] = analysis_preds
analysis_dtfm["Predicted_Label"] = analysis_dtfm["Predicted_Activity"].map({1: "Active", 0: "Inactive"})
analysis_dtfm["Confidence_Active"] = analysis_probs[:, 1]

# Max Tanimoto similarity to active train molecules
from rdkit import DataStructs
train_actives = train_dtfm[train_dtfm["Activity_values"] == 1]
train_active_fps = train_actives["FP_object"]
analysis_fps = analysis_dtfm["FP_object"]

def max_tanimoto_to_actives(fp):
    return max(DataStructs.TanimotoSimilarity(fp, active_fp) for active_fp in train_active_fps)

analysis_dtfm["Max_Tanimoto_to_Active"] = analysis_fps.apply(max_tanimoto_to_actives)

# Final results
analysis_dtfm[["CHEMBL ID", "Predicted_Label", "Confidence_Active", "Max_Tanimoto_to_Active"]]
