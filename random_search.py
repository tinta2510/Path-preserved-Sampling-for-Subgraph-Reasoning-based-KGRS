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
        "lr": sample_log_uniform(1e-4, 1e-3),
        "hidden_dim": random.choice([32, 48, 64, 128]),
        "n_layer": random.choice([3, 4, 5]),
        "dropout": random.uniform(0.0, 0.4),
        "Gumbel_tau": random.uniform(0.5, 1.5),
        "K": random.choice([30, 50, 60, 70, 80]),
        "lamb": sample_log_uniform(5e-5, 1e-3),
    }

# =========================
# Random search loop
# =========================
N_TRIALS = 30
DATA_PATH = "data/last-fm-lightkg/"
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
    ]

    start = time.time()
    subprocess.run(cmd)
    runtime = time.time() - start

    cfg["trial"] = trial
    cfg["runtime_sec"] = runtime

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(cfg) + "\n")
