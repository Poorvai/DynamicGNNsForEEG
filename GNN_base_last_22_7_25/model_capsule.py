import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, GINConv, GINEConv
from torch_geometric.utils import add_self_loops
from moment_adjacency import build_edge_index_and_weights
from GraphCapsule import GraphCapsuleConv
from scipy.stats import skew, kurtosis

def compute_node_statistics(window):
    """
    Compute mean, variance, skewness, and kurtosis for each EEG channel.

    Args:
        window (Tensor): Shape [15, 10000], EEG time-series data.

    Returns:
        Tensor: Shape [15, 4], where each row is [mean, var, skew, kurt] for a channel.
    """
    if isinstance(window, torch.Tensor):
        window = window.detach().cpu().numpy()

    stats = []
    for channel in window:
        mean_val = channel.mean()
        var_val = channel.var()
        skew_val = skew(channel)
        kurt_val = kurtosis(channel, fisher=True)  # fisher=True gives 0 for normal dist
        stats.append([mean_val, var_val, skew_val, kurt_val])

    return torch.tensor(stats, dtype=torch.float32)


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
                 GNN_type = 'GraphCapsuleConv'
                 ):
        super(EEGGraphModel, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.adj_type = Adj_type
        self.gnn_type = GNN_type
        
        self.gnn1 = GraphCapsuleConv(
                input_dim=1, 
                hidden_dim=gnn_hidden_dim, 
                num_gfc_layers=1, 
                num_stats_in=4, 
                num_stats_out=1
            )

        # Final classifier
        self.classifier = nn.Linear(gnn_hidden_dim, num_classes)

    def forward(self, data):
        """
        Args:
            data: Tensor of shape [num_channels, num_timesteps]
        Returns:
            logits: [1, num_classes]
        """

        
        if self.adj_type == 'corr':
            edge_index, edge_weight = compute_pearson_graph(data, threshold=0.6)

            # Add self-loops manually
            num_nodes = data.size(0)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            self_loop_weight = torch.ones(num_nodes, device=data.device) * 1.0
            edge_weight = torch.cat([edge_weight, self_loop_weight], dim=0)

        if self.adj_type == 'stat':
            
            edge_index, edge_weight = build_edge_index_and_weights(data)


        # GNN forward pass
        if self.gnn_type == 'GraphCapsuleConv':
            N = data.size(0)
            A = torch.sparse_coo_tensor(edge_index, edge_weight, size=(N, N))   
            x = compute_node_statistics(data)
            x = x.to(data.device)
            h = F.relu(self.gnn1(x, A))
            h = self.dropout(h)

        # Global pooling (batch = all nodes from one graph)
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        graph_embedding = global_add_pool(h, batch)

        return self.classifier(graph_embedding)
