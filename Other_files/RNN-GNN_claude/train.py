import os
import scipy.io
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model import EEGGraphModel


# interictal_path = '/Users/poorvaichandrasen/Downloads/Patient_1'
# preictal_path = '/Users/poorvaichandrasen/Downloads/Patient_1'

# def load_mat_file(filepath):
#     mat = scipy.io.loadmat(filepath)
    
#     # Find the main variable — usually there's only one key not starting with "__"
#     keys = [k for k in mat.keys() if not k.startswith("__")]
#     assert len(keys) == 1, f"Expected 1 main key, got {keys}"
    
#     matrix = mat[keys[0]]  # e.g., matrix = mat['dataStruct']
    
#     # Fix: Extract EEG data properly
#     signal = matrix[0][0][0][:, :10000]  # This is where your actual EEG matrix lives
    
#     # Ensure it's float32 for PyTorch
#     signal = np.array(signal, dtype=np.float32)
    
#     return signal  # [num_channels, num_samples]


# def load_eeg_dataset(interictal_path, preictal_path, max_interictal=1, max_preictal=0):
#     X = []
#     y = []

#     # Load interictal
#     for i in tqdm(range(1, max_interictal + 1), desc="Loading Interictal"):
#         fname = f"Patient_1_interictal_segment_{i:04d}.mat"
#         fpath = os.path.join(interictal_path, fname)
#         if os.path.exists(fpath):
#             signal = load_mat_file(fpath)
#             X.append(signal)
#             y.append(0)  # 0 = interictal

#     # Load preictal
#     for i in tqdm(range(1, max_preictal + 1), desc="Loading Preictal"):
#         fname = f"Patient_1_preictal_segment_{i:04d}.mat"
#         fpath = os.path.join(preictal_path, fname)
#         if os.path.exists(fpath):
#             signal = load_mat_file(fpath)
#             X.append(signal)
#             y.append(1)  # 1 = preictal

#     return X, y  # list of [15, T] arrays and corresponding labels


# class EEGDataset(Dataset):
#     def __init__(self, X, y, normalize=True):
#         self.X = X
#         self.y = y
#         self.normalize = normalize

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         data = self.X[idx].astype(np.float32)  # [15, T]
#         label = self.y[idx]

#         if self.normalize:
#             data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)

#         return data


# # Load EEG data
# X, y = load_eeg_dataset(interictal_path, preictal_path)

# print(X)


# # # Instantiate model, loss, optimizer
# # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# # model = EEGGraphModel().to(device)
# # criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-3)

# # # Training loop
# # from sklearn.metrics import accuracy_score

# # epochs = 5
# # for epoch in range(epochs):
# #     model.train()
# #     total_loss = 0
# #     all_preds, all_labels = [], []

# #     for signals, label in dataloader:
# #         signals = signals.squeeze(0).to(device)  # [15, T]
# #         label = label.to(device)

# #         optimizer.zero_grad()
# #         output = model(signals)  # [1, 2]
# #         loss = criterion(output, label)  # criterion = nn.CrossEntropyLoss()
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()
# #         pred = torch.argmax(output, dim=1).cpu().item()
# #         all_preds.append(pred)
# #         all_labels.append(label.cpu().item())
        
# #         # ✅ Add detailed debug info here:
# #         print(f"    Output: {output.detach().cpu().numpy()}, Label: {label.item()}, Pred: {pred}, Loss: {loss.item():.4f}")

# #     acc = accuracy_score(all_labels, all_preds)
# #     print(f"Epoch {epoch+1}: Total Loss = {total_loss:.4f}, Accuracy = {acc:.4f}\n")


data = scipy.io.loadmat('/Users/poorvaichandrasen/Downloads/Patient_2/Patient_2_test_segment_0001.mat')
print(data)