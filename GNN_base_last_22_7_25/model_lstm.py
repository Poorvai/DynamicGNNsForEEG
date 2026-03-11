import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data

class EEGGraphModel(nn.Module):
    def __init__(self, 
                 input_timesteps=10000, 
                 cnn_out_dim=8, 
                 lstm_hidden_dim=16, 
                 gnn_hidden_dim=16, 
                 num_classes=2):
        super(EEGGraphModel, self).__init__()

        # CNN for reducing temporal resolution and extracting features
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=11, stride=5, padding=5),    # [C, 8, T/5]
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, stride=5, padding=3),    # [C, 16, T/25]
            nn.ReLU(),
            nn.Conv1d(16, cnn_out_dim, kernel_size=5, stride=5, padding=2),  # [C, 32, T/125]
            nn.ReLU()
        )

        # LSTM for temporal modeling (reduced time length)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
        )

        # GNN layers
        self.gnn1 = GCNConv(lstm_hidden_dim, gnn_hidden_dim)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)

        # Final classifier
        self.classifier = nn.Linear(gnn_hidden_dim, num_classes)

    def forward(self, data, GNN_type = 'corr'):  # data: [num_channels, num_timesteps]
        x = data.unsqueeze(1).float()  # [C, 1, T]
        x = self.cnn(x)                # [C, 32, T']
        x = x.permute(0, 2, 1)         # [C, T', 32] → LSTM format

        # LSTM: get final hidden state for each channel
        _, (h_n, _) = self.lstm(x)     # h_n: [1, C, lstm_hidden_dim]
        h_n = torch.tanh(h_n.squeeze(0))  # [C, lstm_hidden_dim]
        

        # Graph construction (correlation-based)
        if GNN_type == 'corr':
            corr = torch.corrcoef(h_n)
            edge_index = (corr.abs() > 0.6).nonzero(as_tuple=False).t().contiguous()
            edge_index = edge_index.long().to(h_n.device)

        

        # GNN forward pass
        h = F.relu(self.gnn1(h_n, edge_index))
        h = F.relu(self.gnn2(h, edge_index))

        # Graph-level representation (mean pooling)
        graph_embedding = global_add_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))

        return self.classifier(graph_embedding)
