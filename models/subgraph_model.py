import torch
import torch.nn as nn
from torch_scatter import scatter

from message import GRUMessageFunction
from aggregator import FullPNAAggregator, SimplifiedPNAAggregator
from pruner import GumbelNodePruner

class AdaptiveSubgraphLayer(nn.Module):
    """
    One message-passing layer for user-centric subgraph reasoning.

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
        pruner_hidden_dim=64,
        use_full_pna=True,
        act=lambda x: x,
        PNA_delta=None,
        Gumbel_tau=None,
        Gumbel_threshold=None,
    ):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_rel = n_rel
        self.n_node = n_node
        self.node_dim = node_dim
        self.act = act

        # Relation embeddings (dimension = node_dim)
        self.rela_embed = nn.Embedding(2 * n_rel + 1 + 2, node_dim)

        # Per-layer message, aggregation, pruning modules
        self.message_fn = GRUMessageFunction(node_dim=node_dim, rel_dim=node_dim)
        if use_full_pna:
            self.aggregator = FullPNAAggregator(node_dim=node_dim, delta=PNA_delta)
        else:
            self.aggregator = SimplifiedPNAAggregator(node_dim=node_dim)
        self.pruner = GumbelNodePruner(
            user_dim=node_dim,
            node_dim=node_dim,
            hidden_dim=pruner_hidden_dim,
        )

    def forward(self, q_sub, q_rel, hidden, edges, nodes, id_layer, n_layer, old_nodes_new_idx):
        """
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
            hidden_new:        [N,D] node states for this layer after gating
            nodes:             [N,2] (possibly modified for last layer)
            final_nodes:       [M,2] nodes used for scoring on the last layer
            old_nodes_new_idx: unchanged mapping (for caller bookkeeping)
            sampled_nodes_idx: [N] boolean mask of nodes kept for h0
            alpha:             [N] node-wise gates α_z
            edges:             [E,6] (possibly filtered on last layer)
        """
        device = hidden.device

        # By default, we keep all nodes for GRU state propagation
        sampled_nodes_idx = torch.ones(nodes.size(0), dtype=torch.bool, device=device)

        # ---- Last layer: keep only item nodes/edges for scoring ----
        if id_layer == n_layer - 1:
            sampled_nodes_idx = (
                torch.gt(nodes[:, 1], self.n_user - 1)
                & torch.lt(nodes[:, 1], self.n_user + self.n_item)
            )

            item_tail_index = (
                torch.gt(edges[:, 3], self.n_user - 1)
                & torch.lt(edges[:, 3], self.n_user + self.n_item)
            )
            edges = edges[item_tail_index]
            if edges.numel() == 0:
               raise ValueError("No items found in the subgraph at the last layer.")
            else:
                nodes, tail_index = torch.unique(
                    edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True
                )
                edges = torch.cat([edges[:, 0:5], tail_index.unsqueeze(1)], dim=1)
                final_nodes = nodes
        else:
            final_nodes = torch.tensor([0], device=device)

        # ---- Edge indexing wrt previous & current node sets ----
        sub = edges[:, 4]  # previous-layer node index
        rel = edges[:, 2]  # relation id
        obj = edges[:, 5]  # current-layer node index

        # 1) GRU-based message computation
        hs = hidden[sub]          # [E, D]
        hr = self.rela_embed(rel) # [E, D]
        messages = self.message_fn(h_src=hs, rel_emb=hr)  # [E, D]

        # 2) Align prev node states with current node set via self-loops
        N = nodes.size(0)
        D = hidden.size(1)
        h_prev_new = torch.zeros(N, D, device=device)
        h_prev_new.index_copy_(0, old_nodes_new_idx, hidden)

        # 3) PNA-style aggregation
        h_tilde = self.aggregator(
            messages=messages,
            dst_index=obj,
            h_prev=h_prev_new,
        )  # [N, D]

        # 4) Build user representations h_u from user-type nodes
        B = q_sub.size(0)
        h_user = torch.zeros(B, D, device=device)
        node_batch = nodes[:, 0].long()  # [N]

        for b in range(B):
            mask = (node_batch == b) & (nodes[:, 1] < self.n_user)
            if mask.any():
                h_user[b] = h_tilde[mask].mean(dim=0)

        # 5) Gumbel-sigmoid node gating
        alpha, h_gated = self.pruner(
            h_user=h_user,
            h_node=h_tilde,
            node_batch=node_batch,
        )

        hidden_new = self.act(h_gated)

        return hidden_new, nodes, final_nodes, old_nodes_new_idx, sampled_nodes_idx, alpha, edges
    
class AdaptiveSubgraphModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(AdaptiveSubgraphModel, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.n_rel = params.n_rel
        self.n_users = params.n_users
        self.n_items = params.n_items
        self.n_nodes = params.n_nodes
        self.loader = loader

        acts = {"relu": nn.ReLU(), "tanh": torch.tanh, "idd": lambda x: x}
        act = acts[params.act]

        pruner_hidden_dim = getattr(params, "pruner_hidden_dim", self.hidden_dim // 2)
        use_full_pna = getattr(params, "use_full_pna", True)
        PNA_delta = getattr(params, "PNA_delta", None)
        Gumbel_tau = getattr(params, "Gumbel_tau", None)
        Gumbel_threshold = getattr(params, "Gumbel_threshold", None)

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
                    pruner_hidden_dim=pruner_hidden_dim,
                    use_full_pna=use_full_pna,
                    act=act,
                    PNA_delta=PNA_delta,
                    Gumbel_tau=Gumbel_tau,
                    Gumbel_threshold=Gumbel_threshold,
                )
            )
        self.gnn_layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)  # get score

    def forward(self, subs, rels, mode="train"):
        """
        Args:
            subs: list / array of user ids
            rels: list / array of query relation ids (NOTE: currently unused here)
            mode: 'train' or 'test' (passed to loader.get_neighbors)
            return_subgraph_info: kept for API compatibility (ignored for now)

        Returns:
            scores_all: [batch_size, n_items] tensor of item scores
        """
        n = len(subs)

        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()

        # Initial node set: one node per user (batch_idx, user_id)
        nodes = torch.cat(
            [torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], dim=1
        )  # [n, 2]

        # Initial hidden states (all zeros)
        hidden = torch.zeros(n, self.hidden_dim).cuda()

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

        # At this point, `hidden` and `final_nodes` correspond to last-layer nodes
        # (items only, by design of AdaptiveSubgraphLayer's last layer).
        scores = self.W_final(hidden).squeeze(-1)  # [num_last_nodes]

        scores_all = torch.zeros((n, self.n_items)).cuda()
        scores_all[
            [final_nodes[:, 0], final_nodes[:, 1] - self.n_users]
        ] = scores

        return scores_all
