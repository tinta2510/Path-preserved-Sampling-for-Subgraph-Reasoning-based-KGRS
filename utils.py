from typing import List, Tuple, Dict, Set
import random
import math


def load_iter_pairs(iter_path: str) -> List[Tuple[int, int]]:
    """
    Parse train.txt and flatten into (user, pos_item) pairs.

    train.txt format:
        user_id  pos_item_id1  pos_item_id2  ...

    Returns:
        pairs: list of (u, i_pos)
    """
    pairs: List[Tuple[int, int]] = []

    with open(iter_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0])
            items = [int(x) for x in parts[1:]]
            for i in items:
                pairs.append((u, i))

    return pairs
  
def load_positives_dict(iter_path: str) -> Dict[int, Set[int]]:
    """
    test.txt format:
        user_id   pos_item1  pos_item2 ...

    Returns:
        test_pos[u] = set of positive items for user u in test split
    """
    test_pos: Dict[int, Set[int]] = {}

    with open(iter_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0])
            items = [int(x) for x in parts[1:]]

            test_pos[u] = set(items)

    return test_pos
  
def build_user_pos_dict(iter_path: str) -> Dict[int, Set[int]]:
    """
    Build a dict: user_id -> set of positive item_ids
    from train.txt, for fast 'avoid-positives' negative sampling.
    """
    user_pos: Dict[int, Set[int]] = {}

    with open(iter_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0])
            items = [int(x) for x in parts[1:]]

            if u not in user_pos:
                user_pos[u] = set()
            user_pos[u].update(items)

    return user_pos

def sample_negative_item(
    user_id: int,
    num_items: int,
    user_pos_dict: Dict[int, Set[int]],
) -> int:
    """
    Uniform negative sampling over items [0, num_items-1],
    avoiding items the user has interacted with in train.txt.
    """
    pos_items = user_pos_dict.get(user_id, set())
    while True:
        j = random.randint(0, num_items - 1)
        if j not in pos_items:
            return j
        
        
import random
from typing import List, Tuple, Dict, Set
import torch

def train_step_bpr(
    model,                     # UserItemScoringModel
    optimizer: torch.optim.Optimizer,
    user_id: int,
    pos_item_id: int,
    user_pos_dict: Dict[int, Set[int]],
    num_items: int,
    num_negatives: int = 10,
) -> float:
    """
    Perform one BPR training step for a single (user, pos_item) pair.

    Returns:
      loss_value: float
    """
    model.train()

    # 1) Sample negative items for this user
    neg_item_ids = []
    while len(neg_item_ids) < num_negatives:
        j = sample_negative_item(user_id, num_items, user_pos_dict)
        neg_item_ids.append(j)


    # 2) Forward: scores for [pos, neg]
    item_ids = [pos_item_id] + neg_item_ids  # length = 1 + N
    scores, node_states, visited = model(
        user_id=user_id,
        item_ids=item_ids,
        verbose=False,
    )

    s_pos = scores[0]          # scalar
    s_negs = scores[1:]        # [N]
    
    # 3) BPR loss: -log sigma(s_pos - s_neg)
    diffs = s_pos - s_negs          # [N]
    loss_vec = -torch.log(torch.sigmoid(diffs) + 1e-8)  # [N]
    loss = loss_vec.mean()          # scalar
    
    # 4) Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def train_bpr_epochs(
    model,
    optimizer: torch.optim.Optimizer,
    train_pairs: List[Tuple[int, int]],
    user_pos_dict: Dict[int, Set[int]],
    num_items: int,
    num_epochs: int = 2,
    log_every: int = 1000,
    test_pos_dict: Dict[int, Set[int]] = None,  
):
    """
    Train the model with BPR over multiple epochs.

    Args:
      model: UserItemScoringModel
      optimizer: torch optimizer
      train_pairs: list of (user_id, pos_item_id)
      user_pos_dict: user -> set of positive items
      num_items: total number of items
      num_epochs: number of passes over train_pairs
      log_every: how often to print average loss

    Side effect:
      - Updates model parameters in-place
    """
    for epoch in range(num_epochs):
        random.shuffle(train_pairs)

        total_loss = 0.0
        num_steps = 0

        for step, (u, i_pos) in enumerate(train_pairs):
            loss_val = train_step_bpr(
                model=model,
                optimizer=optimizer,
                user_id=u,
                pos_item_id=i_pos,
                user_pos_dict=user_pos_dict,
                num_items=num_items,
            )

            total_loss += loss_val
            num_steps += 1

            if (step + 1) % log_every == 0:
                avg_loss = total_loss / num_steps
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"Step {step+1}/{len(train_pairs)}, "
                    f"avg BPR loss = {avg_loss:.4f}"
                )

        avg_loss = total_loss / max(num_steps, 1)
        print(f"==> Epoch {epoch+1} done. Avg BPR loss: {avg_loss:.4f}")
        
        # ---- Evaluation after epoch ----
        if test_pos_dict is not None:
            metrics = evaluate_model_reachable_only(
                model=model,
                test_pos_dict=test_pos_dict,
                num_items=num_items,
                K_list=[10],
            )
            print(f"==> Evaluation after epoch {epoch+1}: {metrics}")
            
def hit_at_k(ranked_list, positives, k):
    return 1.0 if any(item in positives for item in ranked_list[:k]) else 0.0


def ndcg_at_k(ranked_list, positives, k):
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in positives:
            dcg += 1.0 / math.log2(i + 2)

    # ideal DCG
    ideal_hits = min(len(positives), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_list, positives, k):
    if len(positives) == 0:
        return 0.0
    hit_count = sum(1 for item in ranked_list[:k] if item in positives)
    return hit_count / len(positives)

def evaluate_model_reachable_only(
    model,
    test_pos_dict: Dict[int, Set[int]],
    num_items: int,
    K_list=[10],
):
    """
    Evaluate the model on the test set, but ONLY rank items that are
    reachable in the sampled subgraph of each user.

    Items not in the user's subgraph are never recommended; if a test
    positive is not reachable, it is treated as a miss.
    """
    model.eval()
    all_users = list(test_pos_dict.keys())

    hit_sum = {K: 0.0 for K in K_list}
    ndcg_sum = {K: 0.0 for K in K_list}
    recall_sum = {K: 0.0 for K in K_list}
    count = 0
    reach_sum = 0.0      # total ratio over users
    reach_count = 0
    subgraph_size_sum = 0

    with torch.no_grad():
        for u in all_users:
            positives = test_pos_dict[u]
            if len(positives) == 0:
                continue

            # --- 1) Run propagation once to get subgraph ---
            # We directly use the propagator; no scoring yet.
            node_states, visited = model.propagator.propagate(
                user_id=u,
                num_layers=model.num_layers,
                verbose=False,
            )

            # --- 2) Candidate items: only items in visited subgraph ---
            candidate_items = [i for i in visited if i < num_items]

            # If no items are reachable, this user contributes 0 to metrics
            if len(candidate_items) == 0:
                for K in K_list:
                    hit_sum[K] += 0.0
                    ndcg_sum[K] += 0.0
                count += 1
                continue

            # --- 3) Score only reachable items ---
            scores_list = []
            for i in candidate_items:
                z_i = node_states[i]  # reachable item, so state must exist
                s = model.scorer(z_i)  # [1]
                scores_list.append(s)

            scores = torch.cat(scores_list, dim=0).view(-1)  # [|C_u|]

            # --- 4) Rank reachable items ---
            ranked_indices = torch.argsort(scores, descending=True)
            ranked_items = [candidate_items[i] for i in ranked_indices.tolist()]

            # --- 5) Compute metrics (positives not in candidate_items are misses) ---
            for K in K_list:
                hit_sum[K] += hit_at_k(ranked_items, positives, K)
                ndcg_sum[K] += ndcg_at_k(ranked_items, positives, K)
                recall_sum[K] += recall_at_k(ranked_items, positives, K)

            reachable_pos = positives.intersection(candidate_items)
            coverage_u = len(reachable_pos) / len(positives)   # reachable positive rate

            reach_sum += coverage_u
            reach_count += 1
            # If Recall@K is low but ReachablePositiveRate is also low, the sampler is the bottleneck.
            # If ReachablePositiveRate is high but hits are low, scoring is the bottleneck. 
            
            subgraph_size_sum += len(visited)
            count += 1
            
            

    metrics = {}
    for K in K_list:
        metrics[f"Hit@{K}"] = hit_sum[K] / max(count, 1)
        metrics[f"NDCG@{K}"] = ndcg_sum[K] / max(count, 1)
        metrics[f"Recall@{K}"] = recall_sum[K] / max(count, 1)
        metrics[f"Coverage"] = reach_sum / max(reach_count, 1)
        metrics[f"SubgraphSize"] = subgraph_size_sum / max(count, 1)
        
    return metrics