from typing import List, Tuple, Dict, Set

import torch
import torch.nn as nn
 
from graph_utils import Graph, GraphMeta

class SampledPropagator(nn.Module):
    """
    Two-stage, task-aware, relation-aware propagation:

      - Node type embeddings (item/entity/target-user/other-user)
      - Relation embeddings
      - Edge-level GRU([rel_emb; type_emb], h_s)
      - f_expand(h_s): choose Top-M expanders from frontier
      - f_neighbor(g_{s->v}): choose Top-K neighbors per expander
      - Self-loop for expanders implemented as identity message h_s
      - Only nodes in N_u^{(l+1)} get updated this layer
      - Others carry h^{(l+1)} = h^{(l)}
    """

    def __init__(
        self,
        graph: Graph,
        hidden_dim: int,
        M: int,
        K: int,
        agg: str = "logsumexp",
        device: str = "cpu",
    ):
        super().__init__()
        self.graph = graph
        self.meta = graph.meta
        self.hidden_dim = hidden_dim
        self.M = M
        self.K = K
        self.agg = agg
        self.device = torch.device(device)

        # Node type embeddings: 4 types (item, entity, target user, other user)
        self.type_emb = nn.Embedding(4, hidden_dim)

        # Relation embeddings
        self.rel_emb = nn.Embedding(self.meta.num_relations, hidden_dim)

        # Edge-level GRU: input = [rel_emb ; type_emb], hidden = h_s
        self.gru = nn.GRUCell(input_size=hidden_dim * 2,
                              hidden_size=hidden_dim)

        # f_expand: h_s -> scalar
        self.expand_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # f_neighbor: g_{s->v} -> scalar
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.to(self.device)

    def user_to_node(self, user_id: int) -> int:
        return self.meta.user_node_offset + user_id

    def _node_type_id(self, node_id: int, target_user_id: int) -> int:
        """
        Map node_id -> {0=item, 1=entity, 2=target_user, 3=other_user}
        """
        m = self.meta
        if node_id < m.num_items:
            return 0
        elif node_id < m.num_entities:
            return 1
        elif node_id == m.user_node_offset + target_user_id:
            return 2
        else:
            return 3

    def _agg_messages(self, messages: torch.Tensor) -> torch.Tensor:
        """
        messages: [num_parents, hidden_dim]
        """
        if messages.size(0) == 1:
            return messages[0]
        if self.agg == "mean":
            return messages.mean(dim=0)
        elif self.agg == "logsumexp":
            return torch.logsumexp(messages, dim=0)
        else:
            raise ValueError(f"Unknown agg: {self.agg}")

    @staticmethod
    def _node_type(node_id: int, meta: GraphMeta, target_user_id=None) -> str:
        if node_id < meta.num_items:
            return "item"
        elif node_id < meta.num_entities:
            return "entity"
        elif node_id < meta.num_nodes:
            if target_user_id is not None and node_id == meta.user_node_offset + target_user_id:
                return "target_user"
            return "user"
        else:
            return "invalid"

    def _print_layer_summary(
        self,
        layer: int,
        S: Set[int],
        F: Set[int],
        E: Set[int],
        N: Set[int],
        meta: GraphMeta,
        target_user_id: int,
        max_examples: int = 5,
    ):
        print(
            f"Layer {layer}: "
            f"frontier={len(F)}, expanders={len(E)}, "
            f"new_nodes={len(N)}, visited={len(S)}"
        )

        frontier_sample = list(F)[:max_examples]
        print("  Frontier nodes (id:type):")
        for nid in frontier_sample:
            print(f"    {nid} : {self._node_type(nid, meta, target_user_id)}")

        exp_sample = list(E)[:max_examples]
        print("  Expander nodes (id:type):")
        for nid in exp_sample:
            print(f"    {nid} : {self._node_type(nid, meta, target_user_id)}")

        new_sample = list(N)[:max_examples]
        print("  New nodes N (id:type):")
        for nid in new_sample:
            print(f"    {nid} : {self._node_type(nid, meta, target_user_id)}")

    def propagate(
        self,
        user_id: int,
        num_layers: int,
        verbose: bool = True,
    ):
        """
        Run L-layer *sampled* propagation for a single user.

        Returns:
          - node_states: dict[node_id] -> h_v^{(L)} (torch.Tensor [hidden_dim])
          - S: visited nodes S_u^{(L)} (set of node ids)
        """
        meta = self.meta
        root = self.user_to_node(user_id)
        device = self.device

        node_states: Dict[int, torch.Tensor] = {}
        node_states[root] = torch.zeros(self.hidden_dim, device=device)

        S: Set[int] = {root}  # visited
        F: Set[int] = {root}  # frontier

        if verbose:
            print(f"=== Sampled propagation for user {user_id} (root node {root}) ===")

        for l in range(num_layers):
            if verbose:
                print(f"\n--- Layer {l} ---")
                print(f"Frontier size: {len(F)}, Visited size: {len(S)}")

            if len(F) == 0:
                if verbose:
                    print("Frontier empty; stopping.")
                break

            # -------- Stage A: choose expanders among frontier --------
            frontier_list = list(F)
            h_frontier = []
            for nid in frontier_list:
                h_frontier.append(
                    node_states.get(nid, torch.zeros(self.hidden_dim, device=device))
                )
            h_frontier = torch.stack(h_frontier, dim=0)  # [|F|, d]

            alpha = self.expand_mlp(h_frontier).squeeze(-1)  # [|F|]
            M_eff = min(self.M, len(frontier_list))
            _, topk_idx = torch.topk(alpha, k=M_eff)
            topk_idx = topk_idx.tolist()

            E: Set[int] = {frontier_list[idx] for idx in topk_idx}  # expanders
            C: Set[int] = F.difference(E)                           # carry-over

            if verbose:
                print(f"  Selected {len(E)} expanders, {len(C)} carry-over.")

            # -------- Stage B: neighbors for each expander --------
            msg_dict: Dict[int, List[torch.Tensor]] = {}

            for s in E:
                h_s = node_states.get(
                    s, torch.zeros(self.hidden_dim, device=device)
                )

                dst_nodes, rel_ids = self.graph.neighbors(s)
                dst_nodes = dst_nodes.to(device)
                rel_ids = rel_ids.to(device)

                # Node type embedding for this source node
                type_id = self._node_type_id(s, user_id)
                t_s = self.type_emb(torch.tensor(type_id, device=device))  # [d]

                if dst_nodes.numel() > 0:
                    e_r = self.rel_emb(rel_ids)               # [deg(s), d]
                    t_s_expanded = t_s.unsqueeze(0).expand_as(e_r)  # [deg(s), d]
                    x_in = torch.cat([e_r, t_s_expanded], dim=-1)   # [deg(s), 2d]

                    h_s_expanded = h_s.unsqueeze(0).expand(dst_nodes.size(0), -1)  # [deg(s), d]
                    g = self.gru(x_in, h_s_expanded)            # [deg(s), d]

                    beta = self.neighbor_mlp(g).squeeze(-1)     # [deg(s)]
                    K_eff = min(self.K, dst_nodes.numel())
                    _, topk_eidx = torch.topk(beta, k=K_eff)
                    topk_eidx = topk_eidx.tolist()

                    for eidx in topk_eidx:
                        v = int(dst_nodes[eidx].item())
                        msg = g[eidx]
                        if v not in msg_dict:
                            msg_dict[v] = []
                        msg_dict[v].append(msg)

                # ---- self-loop message (identity) for expander s ----
                if s not in msg_dict:
                    msg_dict[s] = []
                msg_dict[s].append(h_s)

            N: Set[int] = set(msg_dict.keys())

            if verbose:
                print(f"  Nodes receiving messages (N) this layer: {len(N)}")

            # -------- Stage C: update states & sets --------
            for v, msg_list in msg_dict.items():
                msgs = torch.stack(msg_list, dim=0)        # [num_parents, d]
                h_v_new = self._agg_messages(msgs)         # [d]
                node_states[v] = h_v_new

            S |= N
            F = C.union(N)  # expanders stay in frontier via self-loop

            if verbose:
                self._print_layer_summary(l, S, F, E, N, meta, user_id)

            if len(F) == 0:
                if verbose:
                    print("Frontier empty after update; stopping.")
                break

        return node_states, S


class UserItemScoringModel(nn.Module):
    """
    Top-level model:
      - For a given user u, run SampledPropagator to get node states h_v^(L)
      - For each item i, use its node state h_i^(L) (if reached) as z_i
      - Score each item with an MLP: score(u, i) = phi(z_i)

    Personalization is encoded because the propagation / sampling is rooted at u,
    so h_i^(L) depends on u.
    """

    def __init__(
        self,
        graph: Graph,
        hidden_dim: int,
        M: int,
        K: int,
        num_layers: int,
        agg: str = "mean",
        device: str = "cpu",
    ):
        super().__init__()
        self.graph = graph
        self.meta = graph.meta
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Backbone: sampled propagation
        self.propagator = SampledPropagator(
            graph=graph,
            hidden_dim=hidden_dim,
            M=M,
            K=K,
            agg=agg,
            device=device,
        )

        # Scoring head phi(z_i) -> scalar
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.to(self.device)

    def _item_rep(
        self,
        item_id: int,
        node_states: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Item representation z_i:
          - if item node reached in S, use h_i^(L)
          - else, zero vector (no personalized signal)
        """
        if item_id in node_states:
            return node_states[item_id]
        else:
            return torch.zeros(self.hidden_dim, device=self.device)

    def forward(
        self,
        user_id: int,
        item_ids: List[int],
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Set[int]]:
        """
        Compute scores ŷ(u, i) for a single user and a list of items.

        Args:
          user_id: int
          item_ids: list of item indices (0 .. num_items-1)
          verbose: whether to print sampling details

        Returns:
          scores: 1D tensor [len(item_ids)]
          node_states: dict[node_id] -> h_v^(L) (for debug / reuse)
          visited: set of visited node ids S_u^(L)
        """
        # 1) Run sampled propagation rooted at this user
        node_states, visited = self.propagator.propagate(
            user_id=user_id,
            num_layers=self.num_layers,
            verbose=verbose,
        )

        # 2) Item reps + scores
        score_list = []
        for i in item_ids:
            z_i = self._item_rep(i, node_states)      # [d]
            s = self.scorer(z_i)                      # [1]
            score_list.append(s)

        scores = torch.cat(score_list, dim=0).view(-1)  # [len(item_ids)]
        return scores, node_states, visited
