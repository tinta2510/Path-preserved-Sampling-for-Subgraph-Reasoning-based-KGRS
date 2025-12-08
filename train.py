import os
import random
import numpy as np

import torch

from graph_utils import build_graph
from model import UserItemScoringModel

from utils import (
    load_iter_pairs,
    load_positives_dict,
    build_user_pos_dict,
    train_bpr_epochs,
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility in CUDA conv / matmul kernels
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_training():
    # ---------- 1) Paths ----------
    data_dir = "./data/lastfm-lightkg/"  # adjust as needed
    train_path = os.path.join(data_dir, "train.txt")
    test_path  = os.path.join(data_dir, "test.txt")
    kg_path    = os.path.join(data_dir, "kg.txt")

    assert os.path.exists(train_path), f"{train_path} not found"
    assert os.path.exists(test_path),  f"{test_path} not found"
    assert os.path.exists(kg_path),    f"{kg_path} not found"

    # ---------- 2) Build graph ----------
    graph = build_graph(
        train_path=train_path,
        test_path=test_path,
        kg_path=kg_path,
        add_reverse_interaction=True,
    )
    meta = graph.meta
    num_items = meta.num_items

    # ---------- 3) Load training data ----------
    train_pairs = load_iter_pairs(train_path)
    user_pos_dict = build_user_pos_dict(train_path)

    print(f"Total training pairs: {len(train_pairs)}")

    # ---------- 4) Create model ----------
    hidden_dim = 32
    M = 30           # expanders per layer
    K = 15         # neighbors per expander
    num_layers = 4  # propagation depth
    device = "cpu"  # or "cuda" if available

    model = UserItemScoringModel(
        graph=graph,
        hidden_dim=hidden_dim,
        M=M,
        K=K,
        num_layers=num_layers,
        agg="mean",
        device=device,
    )

    # ---------- 5) Optimizer ----------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    # ---------- 6) Train ----------
    num_epochs = 3
    train_bpr_epochs(
        model=model,
        optimizer=optimizer,
        train_pairs=train_pairs,
        user_pos_dict=user_pos_dict,
        num_items=num_items,
        num_epochs=num_epochs,
        log_every=1000,
        test_pos_dict=load_positives_dict(test_path)
    )

    # ---------- 7) Quick sanity check after training ----------
    model.eval()
    test_user = 0
    test_items = list(range(10))
    with torch.no_grad():
        scores, node_states, visited = model(
            user_id=test_user,
            item_ids=test_items,
            verbose=False,
        )

    print(f"\n[After training] Scores for user {test_user} on items {test_items}:")
    for i, s in zip(test_items, scores.tolist()):
        print(f"  item {i}: score = {s:.4f}")
    print(f"Visited nodes |S_u| = {len(visited)}")


if __name__ == "__main__":
    set_seed(42)
    run_training()