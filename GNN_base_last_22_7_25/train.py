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
from utils import load_eeg_dataset, EEGDataset, set_seed

    

interictal_path = preictal_path = '/home/guest/Poorvai/data/'
max_train_file = 1
max_val_file = 1
window = 25000

# Load EEG data
X_train, X_val, y_train, y_val = load_eeg_dataset(interictal_path, preictal_path, window= window, max_train_file= max_train_file, max_val_file=max_val_file)

# Create dataset and dataloader
dataset = EEGDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size = 1 for now
valset = EEGDataset(X_val, y_val)
val_loader = DataLoader(valset, batch_size=1, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device = ', device)
set_seed(0)
model_1 = EEGGraphModel(input_timesteps=window, Adj_type='corr', GNN_type= 'GraphCapsuleConv',
                         cnn_out_dim=8,lstm_hidden_dim=16, gnn_hidden_dim= 12).to(device)
class_weights = torch.tensor([1.2, 1.0]).to(device)
criterion = nn.CrossEntropyLoss()
lr=1e-4
weight_decay=1e-4
optimizer_1 = torch.optim.Adam(model_1.parameters(),lr = lr,weight_decay= weight_decay )


print('Length of Dataset... : ', len(X_train))
print("Criterion used: Cross Entropy")
print('Learning rate: ',lr,' Weight decay = ', weight_decay)
print('TRAINING MODEL TIME = 00:00')
model_start_time = time.time()

 ########################################################################################

# Training loop

epochs = 30
best_val_acc = 0.0
patience = 10
epochs_since_improvement = 0
train_acc_list_1 = []
val_acc_list_1 = []

for epoch in range(epochs):
    start_time = time.time()
    model_1.train()
    total_loss_1 = 0
    all_preds_1, all_labels = [], []

    for signals, label in dataloader:
        signals = signals.squeeze(0).to(device)  # [15, T]
        label = label.to(device)

        optimizer_1.zero_grad()
        output_1 = model_1(signals)  # [1, 2]
        loss_1 = criterion(output_1, label)
        
         
        loss_1.backward()
        optimizer_1.step()
        

        total_loss_1 += loss_1.item()
        pred_1 = torch.argmax(output_1, dim=1).cpu().item()
        all_preds_1.append(pred_1)
        all_labels.append(label.cpu().item())

    train_acc_1 = accuracy_score(all_labels, all_preds_1)
    train_loss_1 = total_loss_1/len(X_train)
    end_time = time.time()

    # Validation
    model_1.eval()
    val_loss_1 = 0.0
    val_preds_1, val_labels = [], []

    with torch.no_grad():
        for signals, label in val_loader:
            signals = signals.squeeze(0).to(device)
            label = label.to(device)

            output_1 = model_1(signals)
            loss_1 = criterion(output_1, label)
            val_loss_1 += loss_1.item()

            pred_1 = torch.argmax(output_1, dim=1).cpu().item()
            val_preds_1.append(pred_1)
            val_labels.append(label.cpu().item())

    val_acc_1 = accuracy_score(val_labels, val_preds_1)

    print(f"Epoch {epoch+1}: "
          f"Model:1 Train Loss = {train_loss_1:.4f}, Train Acc = {train_acc_1:.4f}, "
          f"Val Loss = {val_loss_1/len(X_val):.4f}, Val Acc = {val_acc_1:.4f}, "
          f"Time = {end_time - start_time:.2f}s")

    # Check for improvement
    if val_acc_1 > best_val_acc:
        best_val_acc = val_acc_1
        epochs_since_improvement = 0
        save_path = "/home/guest/Poorvai/GNN_base/saved_model/model_Graphcaps .pth"
        torch.save(model_1.state_dict(), save_path)
        print(f"  ✅ New best model saved at epoch {epoch+1}")
    else:
        epochs_since_improvement += 1
        print(f"  ⚠️ No improvement. Patience counter: {epochs_since_improvement}/{patience}")

    # # Early stopping
    # if epochs_since_improvement >= patience:
    #     print("⏹️ Early stopping triggered.")
    #     break

    train_acc_list_1.append(train_acc_1)
    val_acc_list_1.append(val_acc_1)

model_end_time = time.time()
print("\n\nTraining finished\n\n")
print(f"Model took {(model_end_time - model_start_time)/60:.2f} minutes")

plt.plot(train_acc_list_1, label='train_1', color='cyan')
plt.plot(val_acc_list_1, label='validation_1', color='yellow')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('model accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/guest/Poorvai/GNN_base/Graphcaps.png')
plt.show()