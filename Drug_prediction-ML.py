import pandas as pd
dtfm = pd.read_excel("Molecules.xlsx") 

# Drop rows with missing values
dtfm = dtfm.dropna()

# Check for missing values
print("Missing values per column:")
print(dtfm.isnull().sum())

# Check data types
print("Data types:")
print(dtfm.dtypes)

# Check for duplicates
print("Duplicate SMILES entries:", dtfm["SMILES"].duplicated().sum())
dtfm_clean  = dtfm.drop_duplicates()

# Check SMILES validity
from rdkit import Chem
dtfm_clean["Mol"] = dtfm_clean["SMILES"].apply(Chem.MolFromSmiles)
dtfm_clean = dtfm_clean[dtfm_clean["Mol"].notnull()].reset_index(drop=True)

# Show number of usable molecules
print("Number of molecules after cleaning:", len(dtfm_clean))
dtfm_clean.shape

# Define descriptor functions
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

# Compute each descriptor and add as column
for name, func in descriptor_funcs:
    dtfm_clean[name] = dtfm_clean["Mol"].apply(func)

# Convert activity to numeric values
dtfm_clean["Activity_values"] = dtfm_clean["Activity"].map({"Active": 1, "Inactive": 0})

# Function to compute Morgan fingerprint (2048 bits)
from rdkit.Chem import AllChem
def mol_to_fp(mol, radius=2, nBits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)

# Apply fingerprint function
fps = dtfm_clean["Mol"].apply(mol_to_fp)
dtfm_clean["FP_object"] = fps

# Convert fingerprints to numpy array
import numpy as np
fp_array = np.array([np.array(fp) for fp in fps])

# Create a DataFrame for fingerprints
fp_dtfm = pd.DataFrame(fp_array, columns=[f'FP_{i}' for i in range(fp_array.shape[1])])

# Concatenate with dtfm_clean
dtfm_final = pd.concat([dtfm_clean.reset_index(drop=True), fp_dtfm], axis=1)

# Features to include 
descriptor_cols = ['MolWt', 'LogP', 'NumHAcceptors', 'NumHDonors', 
                   'TPSA', 'NumRotatableBonds', 'RingCount', 'QED']

# Fingerprint columns (FP_0 to FP_2047)
fp_cols = [f'FP_{i}' for i in range(2048)]

# Combine all feature columns
feature_cols = descriptor_cols + fp_cols

# Import necessary scikit-learn tools
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Select molecules from the 'Train' set
train_dtfm = dtfm_final[dtfm_final["Set"] == "Train"]

# Get features and labels
x = train_dtfm[feature_cols]
y = train_dtfm["Activity_values"]

# Split into training and random test set (80% / 20%)
x_train, x_randtest, y_train, y_randtest = train_test_split(
    x, y, test_size=0.2, random_state=188
)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=19)
rf_model.fit(x_train, y_train)

# Evaluate on training set
train_preds = rf_model.predict(x_train)

# Compute confusion matrix
cfmx_train = confusion_matrix(y_train, train_preds)
labels = ['Inactive', 'Active']

# Import tools for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Plot it
plt.figure(figsize=(6, 5))
sns.heatmap(cfmx_train, annot=True, fmt='d', cfmxap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Train Set)')
plt.tight_layout()

# Show training results
print("Training Accuracy:", accuracy_score(y_train, train_preds))
print("\nConfusion Matrix (Train):\n", confusion_matrix(y_train, train_preds))
plt.show()

# Evaluate on random test set
randtest_preds = rf_model.predict(x_randtest)

# Compute confusion matrix
cfmx_randtest = confusion_matrix(y_randtest, randtest_preds)
labels = ['Inactive', 'Active']

# Plot it
plt.figure(figsize=(6, 5))
sns.heatmap(cfmx_randtest, annot=True, fmt='d', cfmxap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Random Test Set)')
plt.tight_layout()

# Show training results
print("Test Accuracy:", accuracy_score(y_randtest, test_preds))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_randtest, randtest_preds))
plt.show()

# Select manual test data
manual_dtfm = dtfm_final[dtfm_final["Set"] == "ManualTest"]

# Get features and labels
x_manual = manual_dtfm[feature_cols]
y_manual = manual_dtfm["Activity_values"]

# Predict using trained model
manual_preds = rf_model.predict(x_manual)

# Evaluate performance
print("Manual Test Accuracy:", accuracy_score(y_manual, manual_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_manual, manual_preds))
print("\nClassification Report:\n", classification_report(y_manual, manual_preds))

# Select the analysis set
analysis_dtfm = dtfm_final[dtfm_final["Set"] == "Analysis"]

# Get features
x_analysis = analysis_dtfm[feature_cols]

# Predict using the trained model
analysis_preds = rf_model.predict(x_analysis)

# Get probability estimates for each analysis prediction
analysis_probs = rf_model.predict_proba(x_analysis)

# Add predictions to the DataFrame
analysis_dtfm["Predicted_Activity"] = analysis_preds
analysis_dtfm["Predicted_Label"] = analysis_dtfm["Predicted_Activity"].map({1: "Active", 0: "Inactive"})

# Column index 1 = probability of being 'Active'
analysis_dtfm["Confidence_Active"] = analysis_probs[:, 1]

# Get RDKit fingerprints of active training molecules
from rdkit import DataStructs
train_actives = train_dtfm[train_dtfm["Activity_values"] == 1]
train_active_fps = train_actives["FP_object"]

# Get RDKit fingerprints of analysis molecules
analysis_fps = analysis_dtfm["FP_object"]

# Function to compute max Tanimoto similarity to active molecules
def max_tanimoto_to_actives(fp):
    return max(DataStructs.TanimotoSimilarity(fp, active_fp) for active_fp in train_active_fps)

# Apply to each analysis fingerprint
analysis_dtfm["Max_Tanimoto_to_Active"] = analysis_fps.apply(max_tanimoto_to_actives)

# Show updated analysis results
analysis_dtfm[["CHEMBL ID", "Predicted_Label", "Confidence_Active", "Max_Tanimoto_to_Active"]]
