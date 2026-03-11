import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class EEGGraphModel(nn.Module):
    def __init__(self, input_timesteps=10000, cnn_out_dim=32, gnn_hidden_dim=32, num_classes=2):
        super(EEGGraphModel, self).__init__()
        # CNN temporal encoder per channel
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=40, stride=10, padding=5),  # [B, 8, T/5]
            nn.ReLU(),
            nn.Conv1d(6, 12, kernel_size=32, stride=8, padding=3),  # [B, 16, T/25]
            nn.ReLU(),
            nn.Conv1d(12, cnn_out_dim, kernel_size=24, stride=6, padding=2),  # [B, out_dim, T/125]
            nn.AdaptiveAvgPool1d(1)  # [B, out_dim, 1]
        )
        self.gnn1 = GCNConv(cnn_out_dim, gnn_hidden_dim)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.classifier = nn.Linear(gnn_hidden_dim, num_classes)

    def forward(self, data):  # data: [num_channels, num_timesteps]
        x = data.unsqueeze(1).float()  # [num_channels, 1, time]
        x = self.encoder(x).squeeze(-1)  # [num_channels, cnn_out_dim]
        
        # Correlation-based graph
        corr = torch.corrcoef(x)
        edge_index = (corr.abs() > 0.33).nonzero(as_tuple=False).t().contiguous()
        edge_index = edge_index.long().to(x.device)
        
        h = F.relu(self.gnn1(x, edge_index))
        h = F.relu(self.gnn2(h, edge_index))
        graph_embedding = global_mean_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        return self.classifier(graph_embedding)
