import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from modules import *  # Assumes get_truth_data, make_sliding_window, TransformerClassifier, reset_weights are defined here

# =======================
# GPU Configuration
# =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =======================
# Load and Preprocess Data
# =======================
TrainingData_dir = "./Phase1Labeled/"
file_dicts = []

for file in os.listdir(TrainingData_dir):
    file_path = os.path.join(TrainingData_dir, file)
    if not os.path.isdir(file_path):
        continue
    truthDataFile = os.path.join(file_path, "TruthData.mat")
    sensorDataFile = os.path.join(file_path, "SensorData.mat")
    file_dicts.append({"file": file, "truthDataFile": truthDataFile, "sensorDataFile": sensorDataFile})

vol = []
y_label = []

for _file_dict in file_dicts:
    sensorDataFile = _file_dict["sensorDataFile"]
    sensorDict = loadmat(sensorDataFile)
    pmu_vm = sensorDict["PMU_Vm"]
    pmu_vm_normalized = MinMaxScaler().fit_transform(pmu_vm)

    label = get_truth_data(_file_dict["truthDataFile"])
    Xseq, yseq = make_sliding_window(pmu_vm_normalized, label, window_size=5, stride=5)

    vol.append(Xseq)
    y_label.append(yseq)

# Convert to PyTorch tensors and move to device
X = torch.tensor(np.array(vol), dtype=torch.float32).to(device)
y = torch.tensor(np.array(y_label), dtype=torch.float32).to(device)

# Dataset details
num_samples = len(X)
num_timesteps = X.shape[1]
num_features = X.shape[2]
num_task = y.shape[1]

# =======================
# Model, Loss, Optimizer
# =======================
input_dim = num_features
hidden_dim = 128
output_dim = num_task
n_heads = 2
n_layers = 2

model = TransformerClassifier(input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps, drop_out=0.3).to(
    device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoader
data = TensorDataset(X, y)
loader = DataLoader(data, batch_size=32, shuffle=True)

# =======================
# Training Loop
# =======================
reset_weights(model)
model.train()

for epoch in range(400):
    epoch_train_loss = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(loader)
    print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}')

# =======================
# Save Model
# =======================
model_path = 'Attack_detector.pth'
torch.save(model.state_dict(), model_path)
print(f'=============\nModel saved to {model_path}\n=============')
print('=============\nFinished training\n=============')
