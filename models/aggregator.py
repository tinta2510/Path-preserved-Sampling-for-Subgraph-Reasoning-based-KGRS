import torch
import torch.nn as nn
from torch_scatter import scatter


class SimplifiedPNAAggregator(nn.Module):
    """
    PNA-style multi-aggregator:

      For each node z at layer ℓ:
        h_mean  = mean({m_{s→z}^ℓ})
        h_max   = max({m_{s→z}^ℓ})
        h_min   = min({m_{s→z}^ℓ})
        h_std   = std({m_{s→z}^ℓ})

      \tilde{h}_z^ℓ = W_agg^{(ℓ)}[h_mean; h_max; h_min; h_std; h_z^{ℓ-1}]

    - Layer-specific linear projection.
    - Assumes message_dim == node_dim (same as GRUMessageFunction).
    """
    def __init__(self, node_dim):
        super().__init__()
        self.node_dim = node_dim
        in_dim = node_dim * 5   # mean, max, min, std, prev h

        self.proj = nn.Linear(in_dim, node_dim)

    def forward(self, messages, dst_index, h_prev):
        """
        Args:
            messages:  [E, node_dim]   edge messages m_{s→z}^ℓ
            dst_index: [E]             destination node indices (0..N-1)
            h_prev:    [N, node_dim]   previous node states h_z^{ℓ-1}

        Returns:
            h_tilde:   [N, node_dim]   aggregated node representations \tilde{h}_z^ℓ
        """
        N = h_prev.size(0)
        D = self.node_dim

        # Counts per node
        ones = torch.ones(messages.size(0), 1, device=messages.device)
        count = scatter(ones, dst_index, dim=0, dim_size=N, reduce="sum")  # [N, 1]
        count_clamped = count.clamp(min=1.0)

        # Basic aggregations
        sum_msg = scatter(messages, dst_index, dim=0, dim_size=N, reduce="sum")     # [N, D]
        mean_msg = sum_msg / count_clamped                                         # [N, D]

        max_msg = scatter(messages, dst_index, dim=0, dim_size=N, reduce="max")    # [N, D]
        min_msg = scatter(messages, dst_index, dim=0, dim_size=N, reduce="min")    # [N, D]

        # Std via E[x^2] - (E[x])^2
        sum_sq = scatter(messages ** 2, dst_index, dim=0, dim_size=N, reduce="sum")
        mean_sq = sum_sq / count_clamped
        var = (mean_sq - mean_msg ** 2).clamp(min=1e-6)
        std_msg = torch.sqrt(var)

        # Concatenate with previous node state and project
        agg_feat = torch.cat([mean_msg, max_msg, min_msg, std_msg, h_prev], dim=-1)  # [N, 5D]
        h_tilde = self.proj(agg_feat)                                      # [N, D]

        return h_tilde

class FullPNAAggregator(nn.Module):
    """
    Full PNA-style aggregator with degree-based scalers.

    Aggregators per node (from incoming messages):
        μ  : mean
        σ  : std  = sqrt(ReLU(μ(x^2) - μ(x)^2) + eps)
        max
        min

    Degree-based scalers:
        S(d, α) = ( log(d + 1) / δ )^α,    α ∈ {0, +1, -1}

    Combined PNA operator (12 channels per node):
        M = [ I
              S(d, α=1)
              S(d, α=-1) ] ⊗ [ μ, σ, max, min ]

    Then:
        h̃_z^ℓ = W_agg^{(ℓ)} [ M_z ; h_z^{ℓ-1} ]
    """
    def __init__(self, node_dim, delta=None, eps=1e-6):
        super().__init__()
        self.node_dim = node_dim
        self.delta = delta  # if None, will be estimated on-the-fly
        self.eps = eps

        in_dim = node_dim * (12 + 1)  # 12 PNA channels + previous h
        self.proj = nn.Linear(in_dim, node_dim)

    def forward(self, messages, dst_index, h_prev):
        """
        Args:
            messages:  [E, D]         edge messages m_{s→z}^ℓ
            dst_index: [E]            destination node indices (0..N-1)
            h_prev:    [N, D]         previous node states h_z^{ℓ-1}
            layer_id:  int            current layer index

        Returns:
            h_tilde:   [N, D]         aggregated node representations h̃_z^ℓ
        """
        N, D = h_prev.size()

        # Node degrees (number of incoming messages)
        ones = torch.ones(messages.size(0), 1, device=messages.device)
        deg = scatter(ones, dst_index, dim=0, dim_size=N, reduce="sum")  # [N, 1]
        deg_clamped = deg.clamp(min=1.0)

        # Base aggregations
        sum_msg = scatter(messages, dst_index, dim=0, dim_size=N, reduce="sum")
        mean_msg = sum_msg / deg_clamped

        max_msg = scatter(messages, dst_index, dim=0, dim_size=N, reduce="max")
        min_msg = scatter(messages, dst_index, dim=0, dim_size=N, reduce="min")

        sum_sq = scatter(messages ** 2, dst_index, dim=0, dim_size=N, reduce="sum")
        mean_sq = sum_sq / deg_clamped
        var = torch.relu(mean_sq - mean_msg ** 2)
        std_msg = torch.sqrt(var + self.eps)

        # Degree-based scalers
        log_deg = torch.log(deg_clamped + 1.0)  # [N, 1]
        if self.delta is None:
            delta = log_deg.mean().detach()
        else:
            delta = torch.tensor(self.delta, device=messages.device, dtype=log_deg.dtype)
        base = (log_deg / delta).clamp(min=1e-6)  # [N, 1]

        scaler_id = torch.ones_like(base)         # α = 0
        scaler_amp = base                         # α = +1
        scaler_att = 1.0 / base                   # α = -1

        # Apply scalers to each aggregator: 3 scalers × 4 aggs = 12 channels
        feats = []
        for agg in (mean_msg, std_msg, max_msg, min_msg):
            feats.append(agg * scaler_id)   # identity
            feats.append(agg * scaler_amp)  # amplifying
            feats.append(agg * scaler_att)  # attenuating

        M = torch.cat(feats, dim=-1)        # [N, 12D]
        agg_feat = torch.cat([M, h_prev], dim=-1)  # [N, 13D]

        h_tilde = self.proj(agg_feat)    # [N, D]
        return h_tilde
