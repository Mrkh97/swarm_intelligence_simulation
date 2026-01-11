"""
Graph Neural Network için robotlar arası iletişim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer
    Multi-head attention mekanizması ile komşu robotlardan bilgi toplar
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Multi-head için boyutlar
        assert out_features % num_heads == 0, (
            "out_features must be divisible by num_heads"
        )
        self.head_dim = out_features // num_heads

        # Weight matrices
        self.W = nn.Linear(in_features, out_features, bias=False)

        # Attention mechanism
        self.attention = nn.Linear(2 * self.head_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch_size, num_nodes, in_features]
            adjacency_matrix: [batch_size, num_nodes, num_nodes] (1 if connected, 0 otherwise)

        Returns:
            output: [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = node_features.shape

        # Linear transformation
        h = self.W(node_features)  # [batch_size, num_nodes, out_features]

        # Reshape for multi-head attention
        h = h.view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Compute attention scores
        # Expand h for broadcasting: [batch, num_nodes, 1, num_heads, head_dim]
        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        # [batch, 1, num_nodes, num_heads, head_dim]
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)

        # Concatenate and compute attention
        h_concat = torch.cat(
            [h_i, h_j], dim=-1
        )  # [batch, num_nodes, num_nodes, num_heads, 2*head_dim]
        h_concat = h_concat.view(
            batch_size, num_nodes, num_nodes, self.num_heads, 2 * self.head_dim
        )

        # Attention coefficients
        e = self.attention(h_concat).squeeze(
            -1
        )  # [batch, num_nodes, num_nodes, num_heads]
        e = self.leaky_relu(e)

        # Mask out non-neighbors using adjacency matrix
        # adjacency_matrix: [batch, num_nodes, num_nodes]
        mask = adjacency_matrix.unsqueeze(-1).expand_as(
            e
        )  # [batch, num_nodes, num_nodes, num_heads]
        e = e.masked_fill(mask == 0, float("-inf"))

        # Softmax to get attention weights
        alpha = F.softmax(e, dim=2)  # [batch, num_nodes, num_nodes, num_heads]
        alpha = self.dropout(alpha)

        # Apply attention to node features
        # alpha: [batch, num_nodes, num_nodes, num_heads]
        # h: [batch, num_nodes, num_heads, head_dim]
        h_expanded = h.unsqueeze(1).expand(
            -1, num_nodes, -1, -1, -1
        )  # [batch, num_nodes, num_nodes, num_heads, head_dim]

        # Weighted sum
        output = torch.sum(
            alpha.unsqueeze(-1) * h_expanded, dim=2
        )  # [batch, num_nodes, num_heads, head_dim]

        if self.concat:
            # Concatenate heads
            output = output.reshape(batch_size, num_nodes, self.out_features)
        else:
            # Average heads
            output = output.mean(dim=2)

        return output


class CommunicationGNN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch_size, num_agents, input_dim] - Her agent'ın local observation'ı
            adjacency_matrix: [batch_size, num_agents, num_agents] - Communication graph

        Returns:
            output: [batch_size, num_agents, output_dim] - GNN ile zenginleştirilmiş features
        """
        # Input projection
        h = self.input_projection(node_features)
        h = F.relu(h)
        h = self.dropout(h)

        # GAT layers with residual connections
        for i, gat_layer in enumerate(self.gat_layers):
            h_new = gat_layer(h, adjacency_matrix)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            # Residual connection
            if i > 0:
                h = h + h_new
                h = self.layer_norm(h)
            else:
                h = h_new

        # Output projection
        output = self.output_projection(h)

        return output


def build_communication_graph(
    positions: np.ndarray,
    communication_range: float,
    num_agents: int,
) -> np.ndarray:
    """
    Agent pozisyonlarından communication graph oluştur

    Args:
        positions: [num_agents, 2] - Agent pozisyonları
        communication_range: float - İletişim mesafesi
        num_agents: int - Agent sayısı

    Returns:
        adjacency_matrix: [num_agents, num_agents] - 1 if connected, 0 otherwise
    """
    adjacency_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)

    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= communication_range:
                    adjacency_matrix[i, j] = 1.0
            else:
                # Self-loop
                adjacency_matrix[i, j] = 1.0

    return adjacency_matrix


def build_communication_graph_batch(
    positions: torch.Tensor,
    communication_range: float,
) -> torch.Tensor:
    """
    Batch processing için communication graph

    Args:
        positions: [batch_size, num_agents, 2]
        communication_range: float

    Returns:
        adjacency_matrix: [batch_size, num_agents, num_agents]
    """
    batch_size, num_agents, _ = positions.shape

    # Compute pairwise distances
    # positions: [batch, num_agents, 2]
    pos_i = positions.unsqueeze(2)  # [batch, num_agents, 1, 2]
    pos_j = positions.unsqueeze(1)  # [batch, 1, num_agents, 2]

    distances = torch.norm(pos_i - pos_j, dim=-1)  # [batch, num_agents, num_agents]

    # Create adjacency matrix
    adjacency_matrix = (distances <= communication_range).float()

    # Ensure self-loops
    eye = torch.eye(num_agents, device=positions.device).unsqueeze(0)
    adjacency_matrix = adjacency_matrix + eye
    adjacency_matrix = (adjacency_matrix > 0).float()

    return adjacency_matrix


if __name__ == "__main__":
    # Test GNN
    batch_size = 32
    num_agents = 5
    input_dim = 100
    hidden_dim = 64

    # Random test data
    node_features = torch.randn(batch_size, num_agents, input_dim)
    positions = torch.randn(batch_size, num_agents, 2) * 10

    # Build graph
    adjacency_matrix = build_communication_graph_batch(
        positions, communication_range=5.0
    )

    # Create GNN
    gnn = CommunicationGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=64,
        num_layers=2,
        num_heads=4,
    )

    # Forward pass
    output = gnn(node_features, adjacency_matrix)

    print(f"Input shape: {node_features.shape}")
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
    print(f"Output shape: {output.shape}")
    print("GNN test passed!")
