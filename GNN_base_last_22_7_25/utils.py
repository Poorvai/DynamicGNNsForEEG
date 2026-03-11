import os
import scipy.io
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model_lstm_corr import EEGGraphModel
import random 
import matplotlib.pyplot as plt
import time

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior for CUDA convolution operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_mat_file(filepath, i, window):

    mat = scipy.io.loadmat(filepath)
    
    # Find the main variable — usually there's only one key not starting with "__"
    keys = [k for k in mat.keys() if not k.startswith("__")]
    assert len(keys) == 1, f"Expected 1 main key, got {keys}"
    matrix = mat[keys[0]]  
    signal = matrix[0][0][0][:, i:i+window]  
    signal = np.array(signal, dtype=np.float32)
    
    return signal  # [num_channels, num_samples]

def load_eeg_dataset(interictal_path, preictal_path, window, max_train_file = 10, max_val_file =8):
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    

    # Load interictal
    for i in tqdm(range(1, max_train_file + 1), desc="Loading Interictal_train"):
        for n in range(2):
            fname = f"Patient_{n+1}_interictal_segment_{i:04d}.mat"
            fpath = os.path.join(interictal_path, fname)
            if os.path.exists(fpath):
                for w in range(0, 2900000, window):
                    signal = load_mat_file(fpath, w, window)
                    X_train.append(signal)
                    y_train.append(0)  # 0 = interictal
    for i in tqdm(range(1, max_val_file+1), desc="Loading Interictal_val"):
        for n in range(2):
            i_val = i + max_train_file+1
            fname = f"Patient_{n+1}_interictal_segment_{i_val:04d}.mat"
            fpath = os.path.join(interictal_path, fname)
            if os.path.exists(fpath):
                for w in range(0, 2900000, window):
                    signal = load_mat_file(fpath, w, window)
                    X_val.append(signal)
                    y_val.append(0)  # 0 = interictal
                

    # Load preictal
    for i in tqdm(range(1, max_train_file + 1), desc="Loading Preictal_train"):
        fname = f"Patient_1_preictal_segment_{i:04d}.mat"
        fpath = os.path.join(preictal_path, fname)
        if os.path.exists(fpath):
            for w in range(0, 2900000,window):
                signal = load_mat_file(fpath, w, window)
                X_train.append(signal)
                y_train.append(1)  # 1 = preictal
    for i in tqdm(range(1, max_val_file + 1), desc="Loading Prerictal_val"):
        i_val = i + max_train_file+1
        fname = f"Patient_1_preictal_segment_{i_val:04d}.mat"
        fpath = os.path.join(preictal_path, fname)
        if os.path.exists(fpath):
            for w in range(0, 2900000, window):
                signal = load_mat_file(fpath, w, window)
                X_val.append(signal)
                y_val.append(1)  # 0 = interictal



    return X_train, X_val, y_train, y_val # list of [15, T] arrays and corresponding labels

class EEGDataset(Dataset):
    def __init__(self, X, y, normalize=True):
        self.X = X
        self.y = y
        self.normalize = normalize

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = self.X[idx].astype(np.float32)  # [15, T]
        label = self.y[idx]

        if self.normalize:
            data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)

        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)


