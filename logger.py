import json
import os
import time
from collections import defaultdict
import numpy as np


class SubgraphLogger:
    """Logger for tracking subgraph statistics and model performance."""
    
    def __init__(self, results_dir='results', dataset_name='model'):
        self.results_dir = results_dir
        self.logs_dir = os.path.join(results_dir, 'logs')
        self.dataset_name = dataset_name
        
        # Create directories if they don't exist
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Storage for aggregating batch statistics
        self.reset_epoch_stats()
    
    def reset_epoch_stats(self):
        """Reset statistics collectors for a new epoch."""
        self.all_subgraph_sizes_before = []
        self.all_subgraph_sizes_after = []
        self.all_num_answer_items_retained = []
        self.all_positive_items = []
        self.all_num_items = []
        self.num_subgraphs = 0
    
    def collect_batch_stats(self, subgraph_info):
        """
        Collect statistics from a single batch.
        
        Args:
            subgraph_info: dict returned from model forward pass
        """
        self.all_subgraph_sizes_before.append(subgraph_info['subgraph_sizes_before_sampling'])
        self.all_subgraph_sizes_after.append(subgraph_info['subgraph_sizes_per_layer'])
        self.all_num_answer_items_retained.extend(subgraph_info['num_answer_items_retained'])
        self.all_positive_items.extend(subgraph_info.get('num_pos_items', []))
        self.all_num_items.extend(subgraph_info['num_items'])
        self.num_subgraphs += subgraph_info["number_of_subgraphs"]
    
    def compute_epoch_summary(self, n_params, train_time, inference_time, recall, ndcg):
        """
        Compute aggregated statistics for the epoch.
        
        Returns:
            dict: Summary statistics
        """
        # Compute average subgraph sizes per layer (before sampling)
        avg_sizes_before_sampling = []
        if self.all_subgraph_sizes_before:
            n_layers = len(self.all_subgraph_sizes_before[0])
            for layer_idx in range(n_layers):
                layer_sizes = [batch_sizes[layer_idx] for batch_sizes in self.all_subgraph_sizes_before]
                avg_sizes_before_sampling.append(np.sum(layer_sizes) / self.num_subgraphs)
        
        # Compute average subgraph sizes per layer (after sampling)
        avg_sizes_after_sampling = []
        if self.all_subgraph_sizes_after:
            n_layers = len(self.all_subgraph_sizes_after[0])
            for layer_idx in range(n_layers):
                layer_sizes = [batch_sizes[layer_idx] for batch_sizes in self.all_subgraph_sizes_after]
                avg_sizes_after_sampling.append(np.sum(layer_sizes) / self.num_subgraphs)
        
        summary = {
            'model_parameters': int(n_params),
            'train_time_seconds': float(train_time),
            'inference_time_seconds': float(inference_time),
            'subgraph_growth_before_sampling': [float(x) for x in avg_sizes_before_sampling],
            'subgraph_growth_after_sampling': [float(x) for x in avg_sizes_after_sampling],
            'avg_positive_items_retained': float(np.sum(self.all_num_answer_items_retained)) / self.num_subgraphs if self.all_num_answer_items_retained else 0.0,
            'avg_positive_items': float(np.sum(self.all_positive_items)) / self.num_subgraphs if self.all_positive_items else 0.0,
            'avg_num_items': float(np.sum(self.all_num_items)) / self.num_subgraphs if self.all_num_items else 0.0,
            'positive_items_retained/subgraph_size': float(np.sum(self.all_num_answer_items_retained)) / (avg_sizes_after_sampling[-1]*self.num_subgraphs) if self.all_num_items else 0.0,
            'positive_items/total_positive_items': float(np.sum(self.all_num_answer_items_retained)) / (np.sum(self.all_positive_items)) if self.all_positive_items else 0.0,
            'ratio_items_retained': float(np.sum(self.all_num_items)) / (avg_sizes_after_sampling[-1]*self.num_subgraphs) if self.all_num_items else 0.0,
            'total_subgraphs_processed': int(self.num_subgraphs),
            'recall@20': float(recall),
            'ndcg@20': float(ndcg),
        }
        
        # Additional statistics
        if self.all_num_answer_items_retained:
            summary['min_positive_items_retained'] = int(np.min(self.all_num_answer_items_retained))
            summary['max_positive_items_retained'] = int(np.max(self.all_num_answer_items_retained))
            summary['std_positive_items_retained'] = float(np.std(self.all_num_answer_items_retained))
        
        if self.all_num_items:
            summary['min_num_items'] = int(np.min(self.all_num_items))
            summary['max_num_items'] = int(np.max(self.all_num_items))
            summary['std_num_items'] = float(np.std(self.all_num_items))
        
        return summary
    
    def save_epoch_log(self, epoch, summary):
        """Save epoch summary to JSON file."""
        log_file = os.path.join(self.logs_dir, f'epoch_{epoch}_summary.json')
        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2)
        return log_file
    
    def save_best_model_log(self, summary):
        """Save best model summary to JSON file."""
        log_file = os.path.join(self.logs_dir, 'best_model_summary.json')
        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2)
        return log_file
    
    def format_summary_string(self, summary):
        """Format summary as human-readable string."""
        out_str = f"[MODEL] Parameters: {summary['model_parameters']:,}\n"
        out_str += f"[TIME] Train: {summary['train_time_seconds']:.2f}s, Inference: {summary['inference_time_seconds']:.2f}s\n"
        out_str += f"[PERFORMANCE] Recall@20: {summary['recall@20']:.4f}, NDCG@20: {summary['ndcg@20']:.4f}\n"
        out_str += f"[SUBGRAPH BEFORE SAMPLING] Growth: {[f'{x:.1f}' for x in summary['subgraph_growth_before_sampling']]}\n"
        out_str += f"[SUBGRAPH AFTER SAMPLING] Growth: {[f'{x:.1f}' for x in summary['subgraph_growth_after_sampling']]}\n"
        out_str += f"[FINAL SUBGRAPH] Avg positive items: {summary['avg_positive_items_retained']:.2f}, "
        out_str += f"Avg total items: {summary['avg_num_items']:.2f}\n"
        return out_str


def count_model_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)