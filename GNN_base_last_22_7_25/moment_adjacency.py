import torch
from torch_geometric.utils import dense_to_sparse, add_self_loops

def compute_node_moments_torch(data):
    """
    Computes 1st to 5th raw moments for each node using PyTorch.
    Input:
        data: torch.Tensor of shape (num_nodes, num_timesteps)
    Output:
        moments: torch.Tensor of shape (num_nodes, 5)
    """
    means = data.mean(dim=1)                     # shape: (num_nodes,)
    centered = data - means.unsqueeze(1)         # shape: (num_nodes, T)
    
    # Raw moments (not normalized by std)
    second = torch.mean(centered ** 2, dim=1)
    third  = torch.mean(centered ** 3, dim=1)
    fourth = torch.mean(centered ** 4, dim=1)
    fifth  = torch.mean(centered ** 5, dim=1)
    
    return torch.stack([means, second, third, fourth, fifth], dim=1)  # shape: (num_nodes, 5)

def compute_squared_distance_matrix_torch(moments):
    """
    Computes pairwise squared Euclidean distances.
    Input:
        moments: torch.Tensor of shape (num_nodes, features)
    Output:
        D: torch.Tensor of shape (num_nodes, num_nodes)
    """
    x = moments
    x_i = x.unsqueeze(1)  # (N, 1, F)
    x_j = x.unsqueeze(0)  # (1, N, F)
    D = torch.sum((x_i - x_j) ** 2, dim=2)  # (N, N)
    return D

def normalize_distance_to_adjacency_torch(D, sigma=None):
    """
    Applies Gaussian kernel to distance matrix.
    Input:
        D: torch.Tensor of shape (N, N)
    Output:
        A: torch.Tensor of shape (N, N), with values in [0, 1]
    """
    if sigma is None:
        sigma = torch.median(D[D > 0])
    A = torch.exp(-D / (2 * sigma ** 2))
    return A

def build_edge_index_and_weights(data, self_loop_weight=1.0):
    """
    Constructs edge_index and edge_weight from EEG data using PyTorch.
    Input:
        data: torch.Tensor of shape (num_nodes, num_timesteps)
    Output:
        edge_index: torch.LongTensor [2, num_edges]
        edge_weight: torch.FloatTensor [num_edges]
    """
    device = data.device
    moments = compute_node_moments_torch(data)                     # (N, 5)
    D = compute_squared_distance_matrix_torch(moments)            # (N, N)
    A = normalize_distance_to_adjacency_torch(D).to(device)       # (N, N)

    edge_index, edge_weight = dense_to_sparse(A)

    # Add self-loops manually
    num_nodes = A.size(0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    self_loop_weights = torch.ones(num_nodes, device=device) * self_loop_weight
    edge_weight = torch.cat([edge_weight, self_loop_weights], dim=0)

    return edge_index, edge_weight
