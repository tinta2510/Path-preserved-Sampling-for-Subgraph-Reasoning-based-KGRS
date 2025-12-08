import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional

import torch

@dataclass
class GraphMeta:
    num_users: int
    num_items: int
    num_entities: int
    num_nodes: int
    num_relations_kg: int
    num_relations: int
    user_node_offset: int
    rel_interact: int
    rel_interact_rev: Optional[int]


@dataclass
class Graph:
    meta: GraphMeta
    node_ptr: torch.LongTensor  # [num_nodes+1]
    edge_dst: torch.LongTensor  # [num_edges]
    edge_rel: torch.LongTensor  # [num_edges]

    def neighbors(self, node_id: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Returns (dst_nodes, rel_ids) for outgoing neighbors of `node_id`.
        """
        start = self.node_ptr[node_id].item()
        end = self.node_ptr[node_id + 1].item()
        return self.edge_dst[start:end], self.edge_rel[start:end]


def _scan_inter_file_for_stats(inter_path: str) -> Tuple[int, int]:
    """
    train.txt format: user_id pos_item_id1 pos_item_id2 ...
    Returns (num_users, num_items) as max_id+1.
    """
    max_user = -1
    max_item = -1

    with open(inter_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0])
            items = [int(x) for x in parts[1:]]
            max_user = max(max_user, u)
            if items:
                max_item = max(max_item, max(items))

    num_users = max_user + 1
    num_items = max_item + 1 if max_item >= 0 else 0
    return num_users, num_items


def _scan_kg_file_for_stats(kg_path: str) -> Tuple[int, int]:
    """
    kg.txt format: ent_id1 rel_id1 ent_id2
    Returns (num_entities, num_relations_kg) as max_id+1.
    """
    max_ent = -1
    max_rel = -1

    with open(kg_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split()
            h = int(h)
            r = int(r)
            t = int(t)
            max_ent = max(max_ent, h, t)
            max_rel = max(max_rel, r)

    num_entities = max_ent + 1
    num_relations_kg = max_rel + 1
    return num_entities, num_relations_kg


def build_graph(train_path: str,
                test_path: str,
                kg_path: str,
                add_reverse_interaction: bool = True) -> Graph:
    """
    Build unified graph from train.txt and kg.txt
    using CSR-like adjacency (node_ptr, edge_dst, edge_rel).
    """

    # ---- 1) Scan stats ----
    num_users_train, num_items_train = _scan_inter_file_for_stats(train_path)
    num_users_test, num_items_test = _scan_inter_file_for_stats(test_path)
    num_users = max(num_users_train, num_users_test)
    num_items = max(num_items_train, num_items_test)
    num_entities, num_relations_kg = _scan_kg_file_for_stats(kg_path)

    assert num_items <= num_entities, (
        f"Expected items to be subrange of entities, "
        f"but num_items={num_items}, num_entities={num_entities}"
    )

    user_node_offset = num_entities
    num_nodes = num_entities + num_users

    # Relation ID layout:
    #   0 .. num_relations_kg-1        : KG forward
    #   num_relations_kg .. 2*R_kg-1   : KG reverse
    #   2*R_kg                         : user->item interaction
    #   2*R_kg + 1                     : item->user interaction (optional)
    rel_interact = num_relations_kg * 2
    rel_interact_rev = rel_interact + 1 if add_reverse_interaction else None

    num_relations = num_relations_kg * 2 + (2 if add_reverse_interaction else 1)

    print("=== Graph stats ===")
    print(f"num_users        = {num_users}")
    print(f"num_items        = {num_items}")
    print(f"num_entities     = {num_entities}")
    print(f"num_nodes        = {num_nodes}")
    print(f"num_relations_kg = {num_relations_kg}")
    print(f"num_relations    = {num_relations}")
    print(f"user_node_offset = {user_node_offset}")
    print(f"rel_interact     = {rel_interact}")
    print(f"rel_interact_rev = {rel_interact_rev}")
    print("====================")

    # ---- 2) Build adjacency lists in Python (per-node lists) ----
    # adj[node] = list of (rel_id, dst_node)
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]

    # 2.1) KG edges (forward + reverse)
    with open(kg_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h_str, r_str, t_str = line.split()
            h = int(h_str)
            r = int(r_str)
            t = int(t_str)

            # forward
            adj[h].append((r, t))

            # reverse
            r_rev = r + num_relations_kg
            adj[t].append((r_rev, h))

    # 2.2) Interaction edges from train.txt
    with open(train_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0])
            items = [int(x) for x in parts[1:]]
            src_user_node = user_node_offset + u

            for i in items:
                # user -> item
                adj[src_user_node].append((rel_interact, i))

                # optional item -> user
                if add_reverse_interaction:
                    adj[i].append((rel_interact_rev, src_user_node))

    # ---- 3) Convert adjacency to CSR-like tensors ----
    # node_ptr: prefix sum of degrees
    degrees = [len(neigh) for neigh in adj]
    num_edges = sum(degrees)

    node_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    for n in range(num_nodes):
        node_ptr[n + 1] = node_ptr[n] + degrees[n]

    edge_dst = torch.empty(num_edges, dtype=torch.long)
    edge_rel = torch.empty(num_edges, dtype=torch.long)

    for n in range(num_nodes):
        start = node_ptr[n].item()
        end = node_ptr[n + 1].item()
        if end > start:
            # fill slice
            rels, dsts = zip(*adj[n])
            edge_rel[start:end] = torch.tensor(rels, dtype=torch.long)
            edge_dst[start:end] = torch.tensor(dsts, dtype=torch.long)

    meta = GraphMeta(
        num_users=num_users,
        num_items=num_items,
        num_entities=num_entities,
        num_nodes=num_nodes,
        num_relations_kg=num_relations_kg,
        num_relations=num_relations,
        user_node_offset=user_node_offset,
        rel_interact=rel_interact,
        rel_interact_rev=rel_interact_rev,
    )

    graph = Graph(
        meta=meta,
        node_ptr=node_ptr,
        edge_dst=edge_dst,
        edge_rel=edge_rel,
    )

    return graph