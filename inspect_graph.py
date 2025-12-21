# inspect.py - Standalone script to analyze fully expanded subgraph sizes
import os
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from load_data import DataLoader

parser = argparse.ArgumentParser(description="Analyze fully expanded subgraph sizes")
parser.add_argument('--data_path', type=str, default='data/last-fm/')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=30, help='Batch size for processing queries')
parser.add_argument('--mode', type=str, default='test', choices=['test', 'train'], help='Which dataset to analyze')

args = parser.parse_args()

class Options(object):
    pass


def compute_fully_expanded_subgraph_sizes(loader, n_layers, batch_size=30, mode='test', device='cuda'):
    """
    Compute the fully expanded subgraph size (without any sampling) for queries.
    
    Args:
        loader: DataLoader instance
        n_layers: Number of GNN layers
        batch_size: Batch size for processing queries
        mode: 'test' or 'train' to select which graph to use
        device: torch device
        
    Returns:
        dict: Statistics about fully expanded subgraph sizes per layer
    """
    # Get queries
    if mode == 'test':
        queries = np.array(loader.test_q)
        n_queries = loader.n_test
    else:
        queries = np.array(loader.train_q)
        n_queries = loader.n_train
    
    n_batch = n_queries // batch_size + (n_queries % batch_size > 0)
    
    # Storage for results
    all_subgraph_sizes = []  # List of dicts with per-query info
    layer_wise_totals = [0] * n_layers  # Total nodes across all queries per layer
    layer_wise_edges = [0] * n_layers   # Total edges across all queries per layer
    
    print(f"\nComputing fully expanded subgraph sizes for {n_queries} {mode} queries...")
    print(f"Number of layers: {n_layers}")
    print(f"Batch size: {batch_size}")
    
    for batch_id in tqdm(range(n_batch), desc="Processing batches"):
        start = batch_id * batch_size
        end = min(n_queries, (batch_id + 1) * batch_size)
        batch_idx = np.arange(start, end)
        
        # Get query subjects
        subs = queries[batch_idx, 0]
        n = len(subs)
        
        # Initial nodes: one per user
        nodes = np.column_stack([
            np.arange(n),  # batch index
            subs           # user id
        ])
        
        batch_subgraph_sizes = []
        batch_edge_counts = []
        
        # Expand layer by layer WITHOUT sampling
        for layer_id in range(n_layers):
            if isinstance(nodes, torch.Tensor):
                nodes = nodes.detach().cpu().numpy()
            nodes, edges, old_nodes_new_idx = loader.get_neighbors(
                nodes=nodes,
                query_users=subs,
                mode=mode
            )
        
            # Record sizes for this layer
            batch_subgraph_sizes.append(nodes.shape[0])
            batch_edge_counts.append(edges.shape[0])
            
            layer_wise_totals[layer_id] += nodes.shape[0]
            layer_wise_edges[layer_id] += edges.shape[0]
        
        # Store aggregate batch info (all queries in batch share same expansion)
        for i in range(n):
            all_subgraph_sizes.append({
                'query_idx': int(start + i),
                'user_id': int(subs[i]),
                'sizes_per_layer': [int(s) for s in batch_subgraph_sizes],
                'edges_per_layer': [int(e) for e in batch_edge_counts],
            })
    
    # Compute statistics
    avg_sizes_per_layer = [total / n_queries for total in layer_wise_totals]
    avg_edges_per_layer = [total / n_queries for total in layer_wise_edges]
    
    # Compute growth ratios
    growth_ratios = []
    for i in range(1, n_layers):
        if avg_sizes_per_layer[i-1] > 0:
            growth_ratios.append(avg_sizes_per_layer[i] / avg_sizes_per_layer[i-1])
        else:
            growth_ratios.append(0.0)
    
    results = {
        'mode': mode,
        'n_queries': n_queries,
        'n_layers': n_layers,
        'avg_subgraph_sizes_per_layer': [float(x) for x in avg_sizes_per_layer],
        'avg_edges_per_layer': [float(x) for x in avg_edges_per_layer],
        'total_nodes_per_layer': [int(x) for x in layer_wise_totals],
        'total_edges_per_layer': [int(x) for x in layer_wise_edges],
        'growth_ratios': [float(x) for x in growth_ratios],
        'final_subgraph_avg_size': float(avg_sizes_per_layer[-1]),
        'final_subgraph_total_size': int(layer_wise_totals[-1]),
    }
    
    return results, all_subgraph_sizes


def save_results(results, all_sizes, results_dir, dataset_name):
    """Save results to JSON files and print summary."""
    logs_dir = os.path.join(results_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save aggregate statistics
    summary_file = os.path.join(logs_dir, f'{dataset_name}_fully_expanded_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Summary to {summary_file}")
    
    # Save per-query details
    details_file = os.path.join(logs_dir, f'{dataset_name}_fully_expanded_details.json')
    with open(details_file, 'w') as f:
        json.dump(all_sizes, f, indent=2)
    print(f"[SAVED] Per-query details to {details_file}")
    
    # Print human-readable summary
    print("\n" + "="*70)
    print(f"FULLY EXPANDED SUBGRAPH STATISTICS ({results['mode'].upper()} SET)")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Number of queries: {results['n_queries']}")
    print(f"Number of layers: {results['n_layers']}")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ Average Subgraph Sizes Per Layer (nodes)                       │")
    print("└─────────────────────────────────────────────────────────────────┘")
    for i, size in enumerate(results['avg_subgraph_sizes_per_layer']):
        print(f"  Layer {i}: {size:>12.1f} nodes")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ Average Edges Per Layer                                        │")
    print("└─────────────────────────────────────────────────────────────────┘")
    for i, edges in enumerate(results['avg_edges_per_layer']):
        print(f"  Layer {i}: {edges:>12.1f} edges")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ Growth Ratios (Layer-to-Layer Expansion)                       │")
    print("└─────────────────────────────────────────────────────────────────┘")
    for i, ratio in enumerate(results['growth_ratios']):
        print(f"  Layer {i} → {i+1}: {ratio:>10.2f}x")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ Final Layer Summary                                            │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print(f"  Average size:  {results['final_subgraph_avg_size']:>10.1f} nodes")
    print(f"  Total size:    {results['final_subgraph_total_size']:>10,} nodes")
    print("="*70 + "\n")
    
    return summary_file, details_file


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Extract dataset name
    dataset = args.data_path.rstrip('/').split('/')
    if len(dataset[-1]) > 0:
        dataset_name = dataset[-1]
    else:
        dataset_name = dataset[-2]
    
    print("="*70)
    print("FULLY EXPANDED SUBGRAPH ANALYSIS")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Data path: {args.data_path}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
        print(f"GPU: {args.gpu}")
    else:
        device = 'cpu'
        print("Device: CPU")
    
    # Configure hyperparameters based on dataset
    opts = Options()
    
    if dataset_name in ['last-fm-lightkg', 'last-fm']:
        opts.n_layer = 3
        opts.K_neg = 20
    elif dataset_name == 'amazon-book':
        opts.n_layer = 3
        opts.K_neg = 20
    elif dataset_name == 'alibaba-fashion':
        opts.n_layer = 3
        opts.K_neg = 20
    else:
        print(f"[WARNING] Unknown dataset '{dataset_name}', using default parameters")
        opts.n_layer = 3
        opts.K_neg = 20
    
    print(f"Layers: {opts.n_layer}")
    print("="*70)
    
    # Load data
    print("\n[LOADING] Data...")
    loader = DataLoader(
        args.data_path, 
        device=device,
        K_neg=opts.K_neg, 
        K_edges=None
    )
    
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.n_users = loader.n_users
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes
    
    print(f"[INFO] Users: {opts.n_users}, Items: {opts.n_items}, "
          f"Entities: {opts.n_ent}, Relations: {opts.n_rel}")
    
    # Run analysis
    results, all_sizes = compute_fully_expanded_subgraph_sizes(
        loader=loader,
        n_layers=opts.n_layer,
        batch_size=args.batch_size,
        mode=args.mode,
        device=device
    )
    
    # Save and display results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    summary_file, details_file = save_results(
        results=results,
        all_sizes=all_sizes,
        results_dir=results_dir,
        dataset_name=dataset_name
    )
    
    print(f"[DONE] Analysis complete!")
    print(f"       Summary: {summary_file}")
    print(f"       Details: {details_file}")