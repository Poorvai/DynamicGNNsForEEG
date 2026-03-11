import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, GINConv, GINEConv
from torch_geometric.utils import add_self_loops
from moment_adjacency import build_edge_index_and_weights
from GraphCapsule import GraphCapsuleConv
import numpy as np
from scipy.signal import coherence
from torch_geometric.utils import dense_to_sparse
from statsmodels.tsa.vector_ar.var_model import VAR

def compute_dtf_graph(data, order=5, threshold=0.1, device='cpu'):
    """
    Computes the adjacency matrix using Directed Transfer Function (DTF).

    Args:
        data (Tensor): [C, T] EEG signal (C: channels/nodes, T: time)
        order (int): Order of the VAR model
        threshold (float): Minimum DTF value for edge inclusion
        device (str or torch.device): Device to place returned tensors on

    Returns:
        edge_index (LongTensor): [2, num_edges] on device
        edge_weight (Tensor): [num_edges] on device
    """
    data_np = data.detach().cpu().numpy().T  # shape: [T, C] for VAR
    num_channels = data_np.shape[1]

    model = VAR(data_np)
    results = model.fit(order)

    # Get coefficient matrices of VAR model
    A_mats = results.coefs  # shape: [order, C, C]
    
    # Frequency domain transform of VAR coefficients (simplified DTF approximation)
    freqs = np.linspace(0, np.pi, 64)
    H = np.zeros((len(freqs), num_channels, num_channels), dtype=np.complex_)

    I = np.eye(num_channels)

    for i, w in enumerate(freqs):
        A_sum = sum(A_mats[k] * np.exp(-1j * w * (k + 1)) for k in range(order))
        H[i] = np.linalg.inv(I - A_sum)

    # Compute DTF
    DTF = np.abs(H) ** 2
    DTF_norm = DTF / np.sum(DTF, axis=2, keepdims=True)

    # Average across frequencies
    dtf_avg = np.mean(DTF_norm, axis=0)

    # Apply threshold and create adjacency matrix
    adj_matrix = (dtf_avg > threshold) * dtf_avg  # shape: [C, C]

    # Convert to PyTorch sparse format
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    edge_index, edge_weight = dense_to_sparse(adj_tensor)

    return edge_index.to(device), edge_weight.to(device)

def compute_coherence_graph(data, fs=250, threshold=0.5):
    """
    Computes adjacency using coherence between EEG channels.

    Args:
        data (Tensor): [C, T] EEG signal (C: channels, T: time)
        fs (int): Sampling frequency
        threshold (float): Coherence threshold for edge inclusion

    Returns:
        edge_index (LongTensor): [2, num_edges]
        edge_weight (Tensor): [num_edges]
    """
    data = data.detach().cpu().numpy()
    num_channels = data.shape[0]
    adj_matrix = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            f, Cxy = coherence(data[i], data[j], fs=fs, nperseg=512)
            mean_coh = np.mean(Cxy)
            if mean_coh > threshold:
                adj_matrix[i, j] = mean_coh
                adj_matrix[j, i] = mean_coh

    # Convert to torch sparse format
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    edge_index, edge_weight = dense_to_sparse(adj_tensor)

    return edge_index, edge_weight

def compute_pearson_graph(h_n, threshold=0.6):
    """
    Compute adjacency graph using Pearson correlation for EEG channels.

    Args:
        h_n: [C, D] tensor of channel embeddings (after LSTM)
        threshold: minimum absolute correlation to consider an edge

    Returns:
        edge_index: [2, num_edges] tensor
        edge_weight: [num_edges] tensor
    """
    C = h_n.size(0)
    
    # Center and normalize each channel vector
    centered = h_n - h_n.mean(dim=1, keepdim=True)
    norm = torch.norm(centered, dim=1, keepdim=True).clamp(min=1e-6)
    normalized = centered / norm

    # Pearson correlation matrix: [C, C]
    corr = torch.matmul(normalized, normalized.T)
    corr = torch.clamp(corr, min=-1.0, max=1.0)

    # Keep edges above threshold
    mask = corr.abs() >= threshold
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    edge_weight = corr[edge_index[0], edge_index[1]]

    # Remove self-loops if you want to add them later yourself
    self_loop_mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, self_loop_mask]
    edge_weight = edge_weight[self_loop_mask]

    # Make edge weights GCN-compatible (non-negative and clipped)
    edge_weight = edge_weight.abs().clamp(min=1e-6, max=0.99)

    return edge_index, edge_weight

class EEGGraphModel(nn.Module):
    def __init__(self, 
                 input_timesteps=10000, 
                 cnn_out_dim=8, 
                 lstm_hidden_dim=16, 
                 gnn_hidden_dim=12, 
                 num_classes=2,
                 Adj_type='corr',
                 GNN_type = 'GCNConv'
                 ):
        super(EEGGraphModel, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.adj_type = Adj_type
        self.gnn_type = GNN_type

        # CNN for temporal downsampling and feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16 , kernel_size=11, stride=5, padding=5),
            nn.ReLU(),
            nn.Conv1d(16, cnn_out_dim, kernel_size=7, stride=25, padding=3),
            nn.ReLU()
        )

        # LSTM to encode temporal features
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )

        # GCN layers (with edge weights)
        if self.gnn_type == 'GCNConv':
            self.gnn1 = GCNConv(lstm_hidden_dim, gnn_hidden_dim)
            self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
            self.classifier = nn.Linear(gnn_hidden_dim, num_classes)

        elif self.gnn_type == 'GINConv':
                self.gnn1 = GINConv(
                    nn=nn.Sequential(
                        nn.Linear(lstm_hidden_dim, gnn_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(gnn_hidden_dim, gnn_hidden_dim)
                    )
                )

                self.gnn2 = GINConv(
                    nn=nn.Sequential(
                        nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(gnn_hidden_dim, gnn_hidden_dim)
                    ))
                
                self.paths = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
                        nn.Linear(gnn_hidden_dim, gnn_hidden_dim)
                    ) for _ in range(5)
                ])
                gnn_hidden_dim = gnn_hidden_dim * 5
                self.classifier = nn.Linear(gnn_hidden_dim, num_classes)


        elif self.gnn_type == 'GraphCapsuleConv':
            
            self.gnn1 = GraphCapsuleConv(
                input_dim=lstm_hidden_dim, 
                hidden_dim=gnn_hidden_dim, 
                num_gfc_layers=2, 
                num_stats_in=1, 
                num_stats_out=5
            )
            gnn_hidden_dim = gnn_hidden_dim * 5
            # self.gnn2 = GraphCapsuleConv(
            #     input_dim=gnn_hidden_dim, 
            #     hidden_dim=gnn_hidden_dim, 
            #     num_gfc_layers=2, 
            #     num_stats_in=1, 
            #     num_stats_out=1
            # )
            self.classifier = nn.Linear(gnn_hidden_dim, num_classes)


        # Final classifier
        

    def forward(self, data):
        """
        Args:
            data: Tensor of shape [num_channels, num_timesteps]
        Returns:
            logits: [1, num_classes]
        """
        x = data.unsqueeze(1).float()         # [C, 1, T]
        x = self.cnn(x)                        # [C, cnn_out_dim, T']
        x = x.permute(0, 2, 1)                 # [C, T', cnn_out_dim]

        # LSTM: extract per-channel representation
        _, (h_n, _) = self.lstm(x)            # h_n: [1, C, lstm_hidden_dim]
        h_n = torch.tanh(h_n.squeeze(0))      # [C, lstm_hidden_dim]

        if self.adj_type == 'corr':
            edge_index, edge_weight = compute_pearson_graph(h_n, threshold=0.3)

            # Add self-loops manually
            num_nodes = h_n.size(0)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            self_loop_weight = torch.ones(num_nodes, device=h_n.device) * 1.0
            edge_weight = torch.cat([edge_weight, self_loop_weight], dim=0)

        if self.adj_type == 'coherence':
            
            edge_index, edge_weight = compute_coherence_graph(data, fs = 5000, threshold= 0.3)
            edge_index = edge_index.to(h_n.device)
            edge_weight = edge_weight.to(h_n.device)



        # GNN forward pass
        if self.gnn_type == 'GraphCapsuleConv':
            N = h_n.size(0)
            A = torch.sparse_coo_tensor(edge_index, edge_weight, size=(N, N))   
            h = F.relu(self.gnn1(h_n, A))
            h = self.dropout(h)

        elif self.gnn_type == 'GCNConv':
            h = F.relu(self.gnn1(h_n, edge_index, edge_weight))
            h = self.dropout(h)
            h = F.relu(self.gnn2(h, edge_index, edge_weight))
            h = self.dropout(h)
        elif self.gnn_type== 'GINConv': 
            h = F.relu(self.gnn1(h_n, edge_index))
            h = self.dropout(h)
            h = F.relu(self.gnn2(h, edge_index))
            h = self.dropout(h)
            h = [path(h) for path in self.paths]
            h = torch.cat(h, dim=-1)

        # Global pooling (batch = all nodes from one graph)
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        graph_embedding = global_add_pool(h, batch)

        return self.classifier(graph_embedding)