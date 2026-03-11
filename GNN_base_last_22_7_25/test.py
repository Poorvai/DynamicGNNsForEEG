import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
from model_lstm_corr import EEGGraphModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import torch.nn.functional as F
mod = 1
def plot_curves(model, test_loader, device):
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for data, label in test_loader:
            data = data.squeeze(0).to(device)
            label = label.item()

            output = model(data)
            prob = F.softmax(output, dim=1)[0,1].item()  # Probability for class 1

            y_true.append(label)
            y_score.append(prob)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color='blue')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid()
    plt.legend()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.subplot(1,2,2)
    plt.plot(recall, precision, color='green')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()

    plt.tight_layout()
    save_path = f'/home/guest/Poorvai/GNN_base/model{mod}.png'
    plt.savefig(save_path)
    plt.show()
    

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for data, label in test_loader:
            data = data.squeeze(0).to(device)
            label = label.to(device).squeeze()

            output = model(data)
            loss = criterion(output, label.unsqueeze(0))

            pred = output.argmax(dim=1).item()

            y_true.append(label.item())
            y_pred.append(pred)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    
    print(f"\nLoss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    return y_true, y_pred


def evaluate_graph_classification(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            probs = torch.softmax(out, dim=1)[:, 1]  # for binary AUC
            preds = out.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_preds)
    y_prob = torch.cat(all_probs)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'auc': roc_auc_score(y_true, y_prob)
    }


device = torch.device('cuda')

interictal_path = preictal_path = '/home/guest/Poorvai/data/'

def load_mat_file(filepath, w):
    mat = scipy.io.loadmat(filepath)
    
    # Find the main variable — usually there's only one key not starting with "__"
    keys = [k for k in mat.keys() if not k.startswith("__")]
    assert len(keys) == 1, f"Expected 1 main key, got {keys}"
    
    matrix = mat[keys[0]]  # e.g., matrix = mat['dataStruct']
    
    # Fix: Extract EEG data properly
    signal = matrix[0][0][0][:, w:w+10000]  # This is where your actual EEG matrix lives
    
    # Ensure it's float32 for PyTorch
    signal = np.array(signal, dtype=np.float32)
    
    return signal  # [num_channels, num_samples]

def load_eeg_dataset(interictal_path, preictal_path, max_interictal=7, max_preictal=7):
    X = []
    y = []

    # Load interictal
    for i in tqdm(range(1, max_interictal + 1), desc="Loading Interictal"):
        fname = f"Patient_1_interictal_segment_{i+13:04d}.mat"
        fpath = os.path.join(interictal_path, fname)
        if os.path.exists(fpath):
            for w in range(0, 2900000,10000):
                signal = load_mat_file(fpath, w)
                X.append(signal)
                y.append(0)  # 1 = preictal

    # Load preictal
    for i in tqdm(range(1, max_preictal + 1), desc="Loading Preictal"):
        fname = f"Patient_1_preictal_segment_{i+13:04d}.mat"
        fpath = os.path.join(preictal_path, fname)
        if os.path.exists(fpath):
            for w in range(0, 2900000,10000):
                signal = load_mat_file(fpath, w)
                X.append(signal)
                y.append(1)  # 1 = preictal

    print('\nSize of dataset = ',len(X), '\n' )
            

    return X, y  # list of [15, T] arrays and corresponding labels

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

# Load EEG data
X, y = load_eeg_dataset(interictal_path, preictal_path, max_interictal= 5, max_preictal=5)

# Create dataset and dataloader
dataset = EEGDataset(X, y)
test_loader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size = 1 for now

print('Length of test set = ',len(X))



def test_model(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.squeeze(0).to(device)   # From [1, 15, 10000] → [15, 10000]
            label = label.to(device).squeeze()  # scalar

            output = model(data)                # output: [1, num_classes]
            loss = criterion(output, label.unsqueeze(0))  # [1, num_classes] vs [1]

            pred = output.argmax(dim=1).item()


            total_loss += loss.item()
            correct += (pred == label.item())
            total += 1

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"\n[Test] Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# Setup
print('Device using: ', device)
# Model

model_path = '/home/guest/Poorvai/GNN_base/saved_model/model.pth'
model = EEGGraphModel( 
                 input_timesteps=10000, 
                 cnn_out_dim=8, 
                 lstm_hidden_dim=16, 
                 gnn_hidden_dim=12, 
                 num_classes=2,
                 Adj_type='coherence',
                 GNN_type = 'GCNConv' )
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device=device)

test_loss, test_acc = test_model(model, test_loader = test_loader, device = device)

# # model_2_path = '/home/guest/Poorvai/GNN_base/saved_model/model_GIN.pth'
# # model_2 = EEGGraphModel( 
# #                  input_timesteps=10000, 
# #                  cnn_out_dim=8, 
# #                  lstm_hidden_dim=16, 
# #                  gnn_hidden_dim=12, 
# #                  num_classes=2,
# #                  Adj_type='corr',
# #                  GNN_type = 'GINConv' )
# # model_2.load_state_dict(torch.load(model_2_path, map_location=device))
# # model_2.to(device=device)

# # test_loss, test_acc = test_model(model_2, test_loader = test_loader, device = device)

model_3_path = '/home/guest/Poorvai/GNN_base/saved_model/model_GraphCapsule_2 .pth'
model_3 = EEGGraphModel( 
                 input_timesteps=10000, 
                 cnn_out_dim=8, 
                 lstm_hidden_dim=16, 
                 gnn_hidden_dim=16, 
                 num_classes=2,
                 Adj_type='coherence',
                 GNN_type = 'GraphCapsuleConv' )
model_3.load_state_dict(torch.load(model_3_path, map_location=device))
model_3.to(device=device)

test_loss, test_acc = test_model(model_3, test_loader = test_loader, device = device)

model_4_path = '/home/guest/Poorvai/GNN_base/saved_model/model_GINConv_MLP .pth'
model_4 = EEGGraphModel( 
                 input_timesteps=10000, 
                 cnn_out_dim=8, 
                 lstm_hidden_dim=16, 
                 gnn_hidden_dim=16, 
                 num_classes=2,
                 Adj_type='coherence',
                 GNN_type = 'GINConv' )
model_4.load_state_dict(torch.load(model_4_path, map_location=device))
model_4.to(device=device)

test_loss, test_acc = test_model(model_4, test_loader = test_loader, device = device)




print('Model 1: Simple GCN')
y_true, y_pred = evaluate_model(model, test_loader, device)
plot_curves(model, test_loader, device)
mod +=1
print('Model 2: GraphCapsule')
y_true, y_pred = evaluate_model(model_3, test_loader, device)
plot_curves(model_3, test_loader, device)
mod +=1
print('Model 3: Modified GIN')
y_true, y_pred = evaluate_model(model_4, test_loader, device)
plot_curves(model_4, test_loader, device)



# model_5_path = '/home/guest/Poorvai/GNN_base/saved_model/model_Graphcaps .pth'
# model_5 = EEGGraphModel( 
#                   input_timesteps=25000, Adj_type='corr', GNN_type= 'GraphCapsuleConv',
#                          cnn_out_dim=8,lstm_hidden_dim=14, gnn_hidden_dim= 14 )
# model_5.load_state_dict(torch.load(model_5_path, map_location=device))
# model_5.to(device=device)
# print('Model 3: Modified GIN')
# y_true, y_pred = evaluate_model(model_5, test_loader, device)
# # plot_curves(model_4, test_loader, device)
