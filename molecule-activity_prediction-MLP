# === Import and clean data ===
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, AllChem
import numpy as np

# Load and clean molecular data
dtfm = pd.read_excel("Molecules.xlsx")
dtfm = dtfm.dropna().drop_duplicates()
dtfm["Mol"] = dtfm["SMILES"].apply(Chem.MolFromSmiles)
dtfm = dtfm[dtfm["Mol"].notnull()].reset_index(drop=True)

# Compute molecular descriptors
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
    dtfm[name] = dtfm["Mol"].apply(func)

dtfm["Activity_values"] = dtfm["Activity"].map({"Active": 1, "Inactive": 0})

# Generate Morgan fingerprints
fps = dtfm["Mol"].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
dtfm["FP_object"] = fps
fp_array = np.array([np.array(fp) for fp in fps])
fp_dtfm = pd.DataFrame(fp_array, columns=[f'FP_{i}' for i in range(fp_array.shape[1])])
dtfm_final = pd.concat([dtfm.reset_index(drop=True), fp_dtfm], axis=1)

# === Prepare features ===
descriptor_cols = ['MolWt', 'LogP', 'NumHAcceptors', 'NumHDonors', 'TPSA', 'NumRotatableBonds', 'RingCount', 'QED']
fp_cols = [f'FP_{i}' for i in range(2048)]
feature_cols = descriptor_cols + fp_cols

# === Prepare training data ===
from sklearn.model_selection import train_test_split
train_df = dtfm_final[dtfm_final["Set"] == "Train"]
x = train_df[feature_cols].values
y = train_df["Activity_values"].values
x_train, x_randtest, y_train, y_randtest = train_test_split(x, y, test_size=0.2, random_state=188)

# Convert to PyTorch tensors
import torch
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_randtest_tensor = torch.tensor(x_randtest, dtype=torch.float32)
y_randtest_tensor = torch.tensor(y_randtest, dtype=torch.float32)
torch.manual_seed(42)

# === Define MLP model ===
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Initialize model, loss function, optimizer
input_size = x_train_tensor.shape[1]
model = MLP(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Training loop ===
num_epochs = 100
loss_values = []
for epoch in range(num_epochs):
    model.train()
    output = model(x_train_tensor).squeeze()
    loss = criterion(output, y_train_tensor)
    loss_values.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# === Plot training loss ===
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 5))
plt.plot(loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Random test set prediction and evaluation ===
model.eval()
with torch.no_grad():
    randtest_outputs = model(x_randtest_tensor).squeeze()
    randtest_preds = (randtest_outputs >= 0.5).float()

y_true = y_randtest
y_probs = randtest_outputs.numpy()
y_pred = randtest_preds.numpy()

from sklearn.metrics import accuracy_score, confusion_matrix
print(f"Random Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")

correct = 0
for i in range(len(y_true)):
    prob = y_probs[i]
    pred = int(y_pred[i])
    actual = int(y_true[i])
    if pred == actual:
        correct += 1
    print(f"Sample {i+1}: Prob = {prob:.2f}, Pred = {pred}, Actual = {actual}, {'Correct' if pred == actual else 'Wrong'}")

print(f"\nCorrect Predictions: {correct}/{len(y_true)}")

# === Confusion matrix ===
import seaborn as sns
cfmx_rand = confusion_matrix(y_true, y_pred)
labels = ["Inactive", "Active"]
plt.figure(figsize=(6, 5))
sns.heatmap(cfmx_rand, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Test Confusion Matrix")
plt.tight_layout()
plt.show()

# === Manual test set prediction ===
manual_df = dtfm_final[dtfm_final["Set"] == "ManualTest"]
x_manual = manual_df[feature_cols].values
y_manual = manual_df["Activity_values"].values
x_manual_tensor = torch.tensor(x_manual, dtype=torch.float32)

model.eval()
with torch.no_grad():
    manual_outputs = model(x_manual_tensor).squeeze()
    manual_preds = (manual_outputs >= 0.5).float()
    manual_probs = manual_outputs.numpy()

y_manual_pred = manual_preds.numpy()
print(f"Manual Test Accuracy: {accuracy_score(y_manual, y_manual_pred):.4f}")

correct = 0
for i in range(len(y_manual)):
    prob = manual_probs[i]
    pred = int(y_manual_pred[i])
    actual = int(y_manual[i])
    if pred == actual:
        correct += 1
    print(f"Sample {i+1}: Prob = {prob:.2f}, Pred = {pred}, Actual = {actual}, {'Correct' if pred == actual else 'Wrong'}")

print(f"\nCorrect Predictions: {correct}/{len(y_manual)}")

# === Confusion matrix for manual test ===
cfmx_manual = confusion_matrix(y_manual, y_manual_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cfmx_manual, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Manual Test Confusion Matrix")
plt.tight_layout()
plt.show()

# === Predict and analyze analysis set ===
analysis_df = dtfm_final[dtfm_final["Set"] == "Analysis"]
x_analysis = analysis_df[feature_cols].values
x_analysis_tensor = torch.tensor(x_analysis, dtype=torch.float32)

model.eval()
with torch.no_grad():
    analysis_outputs = model(x_analysis_tensor).squeeze()
    analysis_probs = analysis_outputs.numpy()
    analysis_preds = (analysis_outputs >= 0.5).float()

analysis_df["Predicted_Activity"] = analysis_preds
analysis_df["Predicted_Label"] = analysis_df["Predicted_Activity"].map({1.0: "Active", 0.0: "Inactive"})
analysis_df["Confidence_Active"] = analysis_probs

# Compute max Tanimoto similarity to active training molecules
from rdkit import DataStructs
train_actives = train_df[train_df["Activity_values"] == 1]
train_active_fps = train_actives["FP_object"]
analysis_fps = analysis_df["FP_object"]

def max_tanimoto_to_actives(fp):
    return max(DataStructs.TanimotoSimilarity(fp, active_fp) for active_fp in train_active_fps)

analysis_df["Max_Tanimoto_to_Active"] = analysis_fps.apply(max_tanimoto_to_actives)

# Show final results
analysis_df[["CHEMBL ID", "Predicted_Label", "Confidence_Active", "Max_Tanimoto_to_Active"]]
