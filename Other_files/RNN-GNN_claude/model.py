
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class EEGGraphModel(nn.Module):
    def __init__(self, input_dim=1, lstm_hidden_dim=8, gnn_hidden_dim=8, num_classes=2):
        super(EEGGraphModel, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        # LSTM for temporal encoding per channel
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True, num_layers=1)
        # GNN layers
        self.gnn1 = GCNConv(lstm_hidden_dim, gnn_hidden_dim)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        # Classifier
        self.classifier = nn.Linear(gnn_hidden_dim, num_classes)
    
    def forward(self, data):  # data: [num_channels, num_timesteps]
        num_channels, num_timesteps = data.shape
        # Step 1: Get LSTM features per channel
        lstm_inputs = data.float().unsqueeze(-1)  # [15, 300000, 1]
        channel_embeddings = []
        for i in range(num_channels):
            out, (h_n, _) = self.lstm(lstm_inputs[i:i+1])  # h_n: [1, 1, hidden_dim]
    
            embedding = h_n.squeeze(0).squeeze(0)          # [hidden_dim]
            embedding = torch.tanh(embedding)              # Apply nonlinearity

            channel_embeddings.append(embedding)
        x = torch.stack(channel_embeddings)  # [num_channels, lstm_hidden_dim]


        # Step 2: Build graph using correlation
        corr = torch.corrcoef(x)  # [num_channels, num_channels]
        edge_index = (corr.abs() > 0.7).nonzero(as_tuple=False).t().contiguous()  # Thresholded edges
        
        # Graph input for PyG
        graph_data = Data(x=x, edge_index=edge_index)
        
        # Step 3: GNN
        h = F.relu(self.gnn1(graph_data.x, graph_data.edge_index))
        h = F.relu(self.gnn2(h, graph_data.edge_index))
        
        # Step 4: Graph pooling and classification
        batch = torch.zeros(num_channels, dtype=torch.long, device=h.device)
        graph_embedding = global_mean_pool(h, batch)  # [1, hidden_dim]
        out = self.classifier(graph_embedding)  # [1, 2]
        return out