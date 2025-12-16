# random_search.py
import subprocess
import random
import math
import json
import os
import time

# =========================
# Search space definitions
# =========================
def sample_log_uniform(low, high):
    return 10 ** random.uniform(math.log10(low), math.log10(high))

def sample_space():
    return {
        "lr": sample_log_uniform(3e-4, 8e-4),
        "hidden_dim": random.choice([48, 64]),
        "n_layer": random.choice([3, 4]),
        "dropout": random.uniform(0.0, 0.2),
        "Gumbel_tau": random.uniform(0.8, 1.5),
        "K": random.choice([110, 130, 150]),
        "lamb": sample_log_uniform(3e-4, 8e-4),
        "item_bonus": random.uniform(0.0, 0.05),
    }

# =========================
# Random search loop
# =========================
N_TRIALS = 10
DATA_PATH = "data/last-fm/"
GPU_ID = 0
LOG_FILE = "results/random_search_log.jsonl"

os.makedirs("results", exist_ok=True)

for trial in range(N_TRIALS):
    cfg = sample_space()

    print(f"\n=== Trial {trial} ===")
    print(cfg)

    cmd = [
        "python", "train_search.py",
        "--data_path", DATA_PATH,
        "--gpu", str(GPU_ID),
        "--lr", str(cfg["lr"]),
        "--lamb", str(cfg["lamb"]),
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--n_layer", str(cfg["n_layer"]),
        "--dropout", str(cfg["dropout"]),
        "--Gumbel_tau", str(cfg["Gumbel_tau"]),
        "--K", str(cfg["K"]),
        "--item_bonus", str(cfg["item_bonus"]),
    ]

    start = time.time()
    subprocess.run(cmd)
    runtime = time.time() - start

    cfg["trial"] = trial
    cfg["runtime_sec"] = runtime

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(cfg) + "\n")
