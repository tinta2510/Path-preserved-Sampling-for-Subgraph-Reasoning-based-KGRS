import torch
import torch.nn as nn


class GumbelNodePruner(nn.Module):
    """
    Task- & semantic-aware node pruning with Gumbel-sigmoid gates.

    For each node z at layer ℓ:
        s_z^ℓ = MLP^{(ℓ)}([h_u^ℓ || h_z^ℓ])
        α_z^ℓ = σ((s_z^ℓ + g) / τ),    g ~ Gumbel(0, 1)

    During training:
        α_z^ℓ is a relaxed gate in (0, 1) used to scale node features.

    During evaluation:
        we use hard gating with threshold θ:
            α_z^ℓ = 1 if s_z^ℓ > θ else 0.
    """
    def __init__(self, user_dim, node_dim, hidden_dim, tau=None, threshold=None):
        super().__init__()
        if tau is None:
            self.tau = 1.1
        else:
            self.tau = tau
            
        if threshold is None:
            self.threshold = 0.3
        else:
            self.threshold = threshold
            
        self.user_dim = user_dim
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold

        in_dim = user_dim + node_dim
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_user, h_node, node_batch):
        """
        Args:
            h_user:    [B, user_dim] user representations at layer ℓ
            h_node:    [N, node_dim] node representations to be scored
            node_batch:[N]           batch index (0..B-1) of each node
            layer_id:  int           current layer index

        Returns:
            alpha:     [N]           gate values α_z^ℓ
            h_gated:   [N, node_dim] gated node representations α_z^ℓ * h_z^ℓ
        """
        # Gather the corresponding user vector for each node
        h_u_for_nodes = h_user[node_batch]  # [N, user_dim]
        x = torch.cat([h_u_for_nodes, h_node], dim=-1)  # [N, user_dim + node_dim]

        s = self.scorer(x).squeeze(-1)  # [N]

        if self.training:
            # Sample Gumbel noise
            uniform = torch.rand_like(s)
            g = -torch.log(-torch.log(uniform.clamp(min=1e-10)).clamp(min=1e-10))
            logits = (s + g) / self.tau
            alpha = torch.sigmoid(logits)
        else:
            # Hard gating at inference
            alpha = (s > self.threshold).float()

        h_gated = h_node * alpha.unsqueeze(-1)
        return alpha, h_gated
