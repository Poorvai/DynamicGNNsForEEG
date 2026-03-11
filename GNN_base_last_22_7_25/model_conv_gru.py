import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

def pairwise_corrcoef(x):
    x = x - x.mean(dim=1, keepdim=True)
    x = x / (x.std(dim=1, keepdim=True) + 1e-6)
    return (x @ x.T) / x.size(1)

class EEGGraphModel(nn.Module):
    def __init__(self, input_timesteps=10000, cnn_out_dim=32, gru_hidden_dim=64, 
                 gnn_hidden_dim=32, num_classes=2):
        super(EEGGraphModel, self).__init__()

        # CNN for temporal feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=11, stride=5, padding=5),   # [B, 8, T/5]
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, stride=5, padding=3),   # [B, 16, T/25]
            nn.ReLU(),
            nn.Conv1d(16, cnn_out_dim, kernel_size=5, stride=5, padding=2),  # [B, cnn_out_dim, T/125]
            nn.ReLU()
        )

        self.gru = nn.GRU(input_size=cnn_out_dim, hidden_size=gru_hidden_dim, 
                          batch_first=True, bidirectional=True)

        self.gnn1 = GCNConv(gru_hidden_dim * 2, gnn_hidden_dim)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.classifier = nn.Linear(gnn_hidden_dim, num_classes)

    def forward(self, data):  # data: [num_channels, num_timesteps]
        x = data.unsqueeze(1).float()  # [C, 1, T]
        x = self.cnn(x)                # [C, cnn_out_dim, T']
        x = x.permute(0, 2, 1)         # [C, T', cnn_out_dim] for GRU

        _, h_n = self.gru(x)           # h_n: [2, C, hidden_dim]
        h_n = h_n.permute(1, 0, 2).reshape(x.size(0), -1)  # [C, 2*hidden_dim]

        # Build correlation-based graph
        corr = pairwise_corrcoef(h_n)
        edge_index = (corr.abs() > 0.7).nonzero(as_tuple=False).t().contiguous()
        edge_index = edge_index.long().to(h_n.device)
        if edge_index.size(1) == 0:
            edge_index = torch.stack([torch.arange(h_n.size(0)), torch.arange(h_n.size(0))], dim=0)

        h = F.relu(self.gnn1(h_n, edge_index))
        h = F.relu(self.gnn2(h, edge_index))
        graph_embedding = global_mean_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        return self.classifier(graph_embedding)

# print("Confusion Matrix:")
# print(confusion_matrix(val_labels, val_preds))