import torch
import torch.nn as nn
from torch_scatter import scatter

from .message import GRUMessageFunction
from .aggregator import FullPNAAggregator, SimplifiedPNAAggregator
from .scorer import GumbelNodeScorer

# ################# HYPERBOLIC OPERATIONS ########################
MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)

def artanh(x):
    return Artanh.apply(x)

def tanh(x):
    return x.clamp(-15, 15).tanh()

def expmap0(u, c=1):
    """Exponential map at origin: tangent space -> hyperbolic space"""
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)

def logmap0(y, c=1):
    """Logarithmic map at origin: hyperbolic space -> tangent space"""
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

def project(x, c=1):
    """Project points to Poincare ball"""
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def mobius_add(x, y, c=1):
    """Mobius addition in Poincare ball"""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)

def hyp_distance(x, y, c=1):
    """Hyperbolic distance in Poincare ball"""
    sqrt_c = c ** 0.5
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * xy + c * y2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c

# ################################################################

class AdaptiveSubgraphLayer(nn.Module):
    """
    One message-passing layer for user-centric subgraph reasoning with hyperbolic embeddings.

    This layer:
      - computes edge messages with a GRU-based message function,
      - aggregates them using a (full) PNA-style aggregator,
      - applies a Gumbel-sigmoid node-wise pruning gate.

    All submodules are per-layer (no dependence on n_layers).
    """
    def __init__(
        self,
        node_dim,
        n_user,
        n_item,
        n_rel,
        n_node,
        use_full_pna=True,
        act=lambda x: x,
        PNA_delta=None,
        Gumbel_tau=None,
        K=50,
        c=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_rel = n_rel
        self.n_node = n_node
        self.node_dim = node_dim
        self.act = act
        self.K = K
        self.device = device
        self.c = c  # Hyperbolic curvature

        # Relation embeddings (initialized in tangent space)
        self.rela_embed = nn.Embedding(2 * n_rel + 1 + 2, node_dim)

        # Per-layer message, aggregation, pruning modules
        self.message_fn = GRUMessageFunction(node_dim=node_dim, rel_dim=node_dim)
        if use_full_pna:
            self.aggregator = FullPNAAggregator(node_dim=node_dim, delta=PNA_delta)
        else:
            self.aggregator = SimplifiedPNAAggregator(node_dim=node_dim)
        self.scorer = GumbelNodeScorer(
            user_dim=node_dim,
            node_dim=node_dim,
            tau=Gumbel_tau
        )

    def forward(self, q_sub, q_rel, hidden, edges, nodes,
                id_layer, n_layer, old_nodes_new_idx):
        """
        Forward pass with hyperbolic message passing.
        hidden is expected to be in tangent space.

        Args:
            q_sub:   [B]         user ids for each query in the batch
            q_rel:   [B]         query relation ids (currently unused here)
            hidden:  [N_prev,D]  node states from previous layer
            edges:   [E,6]       [batch_idx, head, rel, tail, old_idx, new_idx]
            nodes:   [N,2]       [batch_idx, node_id] for current layer
            id_layer: int        current layer index
            n_layer: int         total number of layers
            old_nodes_new_idx: [N_prev] mapping from previous-layer node index
                            to new-layer node index (built via self-loops)

        Returns:
            hidden_new:        [N_kept,D] node states for this layer after gating
            nodes:             [N_kept,2] (pruned node set; last layer = items only)
            final_nodes:       [M,2] nodes used for scoring on the last layer
            old_nodes_new_idx: unchanged mapping (for caller bookkeeping)
            sampled_nodes_idx: [N] boolean mask over the *input* nodes before pruning
            alpha:             [N] node-wise gates α_z (before pruning)
            edges:             [E,6] unchanged (not used by later layers)
        """
        # device = hidden.device

        # ---- Edge indexing wrt previous & current node sets ----
        sub = edges[:, 4]  # previous-layer node index
        rel = edges[:, 2]  # relation id
        obj = edges[:, 5]  # current-layer node index

        # 1) Get embeddings in tangent space and map to hyperbolic space
        hs = hidden[sub]           # [E, D] - in tangent space
        hr = self.rela_embed(rel)  # [E, D] - in tangent space
        
        # Map to hyperbolic space
        hs_hyp = expmap0(hs, self.c)  # [E, D] - in hyperbolic space
        hr_hyp = expmap0(hr, self.c)  # [E, D] - in hyperbolic space
        
        # Hyperbolic message computation: TransE in hyperbolic space
        message_hyp = project(mobius_add(hs_hyp, hr_hyp, self.c), self.c)  # [E, D]
        
        # Map back to tangent space for aggregation
        messages = logmap0(message_hyp, self.c)  # [E, D] - in tangent space

        # 2) Align prev node states with current node set via self-loops
        N = nodes.size(0)
        D = hidden.size(1)
        h_prev_new = torch.zeros(N, D, device=self.device)
        h_prev_new.index_copy_(0, old_nodes_new_idx, hidden)

        # 3) PNA-style aggregation (in tangent space)
        h_tilde = self.aggregator(
            messages=messages,
            dst_index=obj,
            h_prev=h_prev_new,
        )  # [N, D]

        # 4) Build user representations h_u from user-type nodes
        B = q_sub.size(0)
        node_batch = nodes[:, 0].long()      # [N]
        node_ent   = nodes[:, 1].long()      # [N]
        h_user = torch.zeros(B, D, device=self.device)

        for b in range(B):
            center_uid = q_sub[b].item()
            # center user node for this query
            center_mask = (node_batch == b) & (node_ent == center_uid)

            if center_mask.any():
                # there should normally be exactly one; take the first
                h_user[b] = h_tilde[center_mask][0]
            else:
                raise ValueError(
                    f"Center user node (batch {b}, user {center_uid}) not found in current nodes."
                )
                    
        # 5) Gumbel-sigmoid node gating (feature-level)
        alpha, h_gated = self.scorer(
            h_user=h_user,
            h_node=h_tilde,
            node_batch=node_batch,
        )

        # Apply activation in tangent space, then map to hyperbolic space
        hidden_tangent = self.act(h_gated)   # [N, D] - tangent space
        hidden_all = hidden_tangent  # Keep in tangent space for next layer

        # ==================================================================
        # 6) AdaProp-style INCREMENTAL SAMPLING using alpha as score
        # ==================================================================
        N = nodes.size(0)
        keep_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

        # (a) always keep previously selected nodes V^{l-1}
        keep_mask[old_nodes_new_idx] = True

        # (b) identify newly-visited nodes as candidates:
        #     CAND = N(V^{l-1}) \ V^{l-1}
        diff_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        diff_mask[old_nodes_new_idx] = False      # True only for "new" nodes
        candidate_idx = diff_mask.nonzero(as_tuple=False).squeeze(-1)

        # If we are NOT at the last layer and have a top_k budget, sample;
        # for the last layer we can skip sampling and keep all (then filter items).
        if (
            self.K is not None
            and self.K > 0
            and candidate_idx.numel() > 0
            and id_layer < n_layer - 1
        ):
            cand_batch = node_batch[candidate_idx]   # [num_cand]
            cand_alpha = alpha[candidate_idx]        # [num_cand]

            # sample independently per query in the batch
            for b in range(B):
                mask_b = cand_batch == b
                idx_b = candidate_idx[mask_b]
                if idx_b.numel() == 0:
                    continue

                if idx_b.numel() <= self.K:
                    # fewer than K: keep all
                    keep_mask[idx_b] = True
                else:
                    # take top-k by alpha (higher gate = more relevant)
                    _, top_pos = torch.topk(cand_alpha[mask_b], k=self.K)
                    chosen_idx = idx_b[top_pos]
                    keep_mask[chosen_idx] = True
        else:
            # no structural sampling: keep all nodes in this layer
            keep_mask[:] = True

        # (c) for the LAST layer, restrict to item-type nodes only
        if id_layer == n_layer - 1:
            is_item = (nodes[:, 1] >= self.n_user) & (
                nodes[:, 1] < self.n_user + self.n_item
            )
            keep_mask &= is_item
            final_nodes = nodes[keep_mask]
        else:
            final_nodes = None

        sampled_nodes_idx = keep_mask.clone()

        # apply structural pruning
        hidden_new = hidden_all[keep_mask]   # [N_kept, D]
        nodes_new = nodes[keep_mask]         # [N_kept, 2]

        return (
            hidden_new,
            nodes_new,
            final_nodes,
            old_nodes_new_idx,
            sampled_nodes_idx,
            alpha,
            edges,
        )
    
class AdaptiveSubgraphModel(torch.nn.Module):
    def __init__(self, params, loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(AdaptiveSubgraphModel, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.n_rel = params.n_rel
        self.n_users = params.n_users
        self.n_items = params.n_items
        self.n_nodes = params.n_nodes
        self.loader = loader
        self.device = device
        
        acts = {"relu": nn.ReLU(), "tanh": torch.tanh, "idd": lambda x: x}
        act = acts[params.act]

        use_full_pna = getattr(params, "use_full_pna", True)
        PNA_delta = getattr(params, "PNA_delta", None)
        Gumbel_tau = getattr(params, "Gumbel_tau", None)
        K = getattr(params, "K", 50)
        c = getattr(params, "c", 1.0)  # Hyperbolic curvature
        print("Config - use_full_pna:", use_full_pna, " PNA_delta:", PNA_delta, " Gumbel_tau:", Gumbel_tau, " K:", K, " c:", c)

        # Stack per-layer AdaptiveSubgraphLayer modules
        layers = []
        for _ in range(self.n_layer):
            layers.append(
                AdaptiveSubgraphLayer(
                    node_dim=self.hidden_dim,
                    n_user=self.n_users,
                    n_item=self.n_items,
                    n_rel=self.n_rel,
                    n_node=self.n_nodes,
                    use_full_pna=use_full_pna,
                    act=act,
                    PNA_delta=PNA_delta,
                    Gumbel_tau=Gumbel_tau,
                    K=K,
                    c=c,
                    device=self.device,
                )
            )
        self.gnn_layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)  # get score

    def forward(self, subs, rels, mode="train"):
        """
        Forward pass with hyperbolic embeddings.
        Returns item scores in Euclidean space.

        Args:
            subs: list / array of user ids
            rels: list / array of query relation ids (NOTE: currently unused here)
            mode: 'train' or 'test' (passed to loader.get_neighbors)
            return_subgraph_info: kept for API compatibility (ignored for now)

        Returns:
            scores_all: [batch_size, n_items] tensor of item scores
        """
        n = len(subs)

        q_sub = torch.LongTensor(subs).to(self.device)
        q_rel = torch.LongTensor(rels).to(self.device)

        # Initial node set: one node per user (batch_idx, user_id)
        nodes = torch.cat(
            [torch.arange(n).unsqueeze(1).to(self.device), q_sub.unsqueeze(1)], dim=1
        )  # [n, 2]

        # Initial hidden states (all zeros in tangent space)
        hidden = torch.zeros(n, self.hidden_dim).to(self.device)

        final_nodes = None  # will be set on last layer

        for i in range(self.n_layer):
            # Expand user-centric computation graph for this layer
            nodes_np = nodes.data.cpu().numpy()
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(
                nodes_np, mode=mode
            )

            # One AdaptiveSubgraphLayer step
            (
                hidden,
                nodes,
                final_nodes,
                old_nodes_new_idx,
                sampled_nodes_idx,
                alpha,
                edges,
            ) = self.gnn_layers[i](
                q_sub,
                q_rel,
                hidden,
                edges,
                nodes,
                i,
                self.n_layer,
                old_nodes_new_idx,
            )

            hidden = self.dropout(hidden)

        # At this point, `hidden` is in tangent space
        # For scoring, we can use it directly or optionally compute distances
        scores = self.W_final(hidden).squeeze(-1)  # [num_last_nodes]

        scores_all = torch.zeros((n, self.n_items)).to(self.device)
        scores_all[(final_nodes[:, 0], final_nodes[:, 1] - self.n_users)] = scores

        return scores_all
