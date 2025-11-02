import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def get_truth_data(truthDataFile):
    truthDict = loadmat(truthDataFile)
    truth = np.transpose(truthDict["AttackTruth"][0])
    return truth


def make_sliding_window(X, y, window_size, stride):
    X_seq = []
    y_seq = []

    for i in range(0, len(X) - window_size + 1, stride):
        x_window = X[i:i + window_size]  # shape: (window_size, D)
        y_window = y[i:i + window_size]  # shape: (window_size,)

        x_flat = x_window.flatten()  # shape: (5 * D,)

        # Convert labels to one scalar
        y_label = y_window.max()  # if any attack in window

        X_seq.append(x_flat)
        y_seq.append(y_label)

    X_seq = np.stack(X_seq)  # shape: (num_windows, window_size, D)
    y_seq = np.stack(y_seq)  # shape: (num_windows, window_size)
    return X_seq, y_seq

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps,drop_out):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_timesteps, hidden_dim))
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers)
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Flatten the output of the transformer
        self.dropout = nn.Dropout(p=drop_out)
        self.bn = nn.BatchNorm1d(hidden_dim * num_timesteps)

    def forward(self, src):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch, feature)
        transformer_output = self.transformer.encoder(src_emb)
        transformer_output = transformer_output.permute(1, 0, 2).contiguous().view(src.size(0), -1)  # Flatten
        transformer_output = self.bn(transformer_output)
        transformer_output = self.dropout(transformer_output)
        predictions = self.fc(transformer_output)
        return predictions
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def make_sliding_window_single(X, y, window_size, stride):
    X_seq = []
    y_seq = []

    for i in range(0, len(X) - window_size + 1, stride):
        x_window = X[i:i + window_size]  # shape: (window_size, D)
        y_window = y[i:i + window_size]  # shape: (window_size, num_classes)

        x_flat = x_window.flatten()  # shape: (window_size * D,)

        # Sum across time dimension (rows) → vector of class counts
        class_counts = np.sum(y_window, axis=0)  # shape: (num_classes,)

        # Get the index (class) with the most 1's

        X_seq.append(x_flat)
        y_seq.append(class_counts)

    X_seq = np.stack(X_seq)  # shape: (num_windows, window_size * D)
    y_label = np.argmax(y_seq)  # shape: (num_windows,)
    return X_seq, y_label