import torch
import torch.nn as nn


class GRUMessageFunction(nn.Module):
    """
    GRU-based message function:
      m_{s→z}^ℓ = h_s^ℓ = GRU^{(ℓ)}(r_ℓ, h_s^{ℓ-1})

    - One GRUCell per layer.
    - The message dimension == node_dim.
    """
    def __init__(self, node_dim, rel_dim):
        super().__init__()
        self.node_dim = node_dim
        self.rel_dim = rel_dim

        self.gru = nn.GRUCell(rel_dim, node_dim)

    def forward(self, h_src, rel_emb):
        """
        Args:
            h_src:   [E, node_dim]  source node states h_s^{ℓ-1}
            rel_emb: [E, rel_dim]   relation embeddings r_ℓ for each edge
            layer_id: int, current GNN layer index

        Returns:
            messages: [E, node_dim] messages m_{s→z}^ℓ (updated hidden states)
        """
        messages = self.gru(rel_emb, h_src)   # [E, node_dim]
        return messages