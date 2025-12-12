import os
import numpy as np
from collections import Counter, defaultdict


NUM_USERS = 3000
NUM_ITEMS = 6000


DATA_DIR = './data/last-fm/'
OUT_DIR = './data/last-fm-subset/'
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Read user-item interactions
def read_cf(file):
    inter_mat = []
    with open(file, 'r') as f:
        for line in f:
            arr = list(map(int, line.strip().split()))
            u, items = arr[0], arr[1:]
            for i in set(items):
                inter_mat.append((u, i))
    return np.array(inter_mat)

train_cf = read_cf(os.path.join(DATA_DIR, 'train.txt'))
test_cf = read_cf(os.path.join(DATA_DIR, 'test.txt'))
all_cf = np.concatenate([train_cf, test_cf], axis=0)

# 2. Select users and items by binned frequency sampling
def bin_and_sample(ids, counts, n_bins, n_total):
    # Bin by frequency
    values = np.array([counts[i] for i in ids])
    bin_edges = np.percentile(values, np.linspace(0, 100, n_bins+1))
    bin_edges[-1] += 1  # include max
    bins = defaultdict(list)
    for i, v in zip(ids, values):
        for b in range(n_bins):
            if bin_edges[b] <= v < bin_edges[b+1]:
                bins[b].append(i)
                break
            
    # Proportional sampling
    sampled = []
    total_in_bins = sum(len(bins[b]) for b in range(n_bins))
    for b in range(n_bins):
        prop = len(bins[b]) / total_in_bins
        n_sample = int(round(prop * n_total))
        n_sample = min(n_sample, len(bins[b]))
        sampled.extend(np.random.choice(bins[b], n_sample, replace=False))
        
    # If not enough, fill randomly
    while len(sampled) < n_total:
        for b in range(n_bins):
            if len(sampled) < n_total and bins[b]:
                sampled.append(np.random.choice(bins[b]))
    return set(sampled)

# Example usage:
# all_cf = np.array([[user_id, item_id], ...])
user_counts = Counter(all_cf[:,0])
item_counts = Counter(all_cf[:,1])
user_ids = list(user_counts.keys())
item_ids = list(item_counts.keys())

sampled_users = bin_and_sample(user_ids, user_counts, n_bins=10, n_total=NUM_USERS)
sampled_items = bin_and_sample(item_ids, item_counts, n_bins=10, n_total=NUM_ITEMS)

user_map = {u: idx for idx, u in enumerate(sorted(sampled_users))}
item_map = {i: idx for idx, i in enumerate(sorted(sampled_items))}

def filter_and_remap(cf):
    filtered = []
    for u, i in cf:
        if u in user_map and i in item_map:
            filtered.append((user_map[u], item_map[i]))
    return filtered

train_subset = filter_and_remap(train_cf)
test_subset = filter_and_remap(test_cf)

# 3. Write new train.txt and test.txt
def write_cf(cf, file):
    user_items = defaultdict(list)
    for u, i in cf:
        user_items[u].append(i)
    with open(file, 'w') as f:
        for u in sorted(user_items):
            items = ' '.join(map(str, sorted(set(user_items[u]))))
            f.write(f"{u} {items}\n")

write_cf(train_subset, os.path.join(OUT_DIR, 'train.txt'))
write_cf(test_subset, os.path.join(OUT_DIR, 'test.txt'))

# 4. Filter item_list.txt, user_list.txt, entity_list.txt, kg.txt, relation_list.txt
def filter_list_file(infile, keep_set, outfile):
    with open(infile, 'r') as fin, open(outfile, 'w') as fout:
        for idx, line in enumerate(fin):
            if idx in keep_set:
                fout.write(line)

filter_list_file(os.path.join(DATA_DIR, 'user_list.txt'), set(user_map.keys()), os.path.join(OUT_DIR, 'user_list.txt'))
filter_list_file(os.path.join(DATA_DIR, 'item_list.txt'), set(item_map.keys()), os.path.join(OUT_DIR, 'item_list.txt'))

with open(os.path.join(DATA_DIR, 'entity_list.txt'), 'r') as fin:
    entity_lines = fin.readlines()
item_entities = set(item_map.keys())
filter_list_file(os.path.join(DATA_DIR, 'entity_list.txt'), item_entities, os.path.join(OUT_DIR, 'entity_list.txt'))

with open(os.path.join(DATA_DIR, 'kg.txt'), 'r') as fin, open(os.path.join(OUT_DIR, 'kg.txt'), 'w') as fout:
    for line in fin:
        h, r, t = map(int, line.strip().split())
        if h in item_entities or t in item_entities:
            fout.write(line)

import shutil
shutil.copy(os.path.join(DATA_DIR, 'relation_list.txt'), os.path.join(OUT_DIR, 'relation_list.txt'))

print("Binned subset extraction complete. Files written to", OUT_DIR)