import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from logger import SubgraphLogger

parser = argparse.ArgumentParser(description="Parser for Adaptive Subgraph Model")
parser.add_argument('--data_path', type=str, default='data/last-fm/')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
   
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options()
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    if torch.cuda.is_available():   
        torch.cuda.set_device(args.gpu)

    if dataset in ['last-fm-lightkg', 'last-fm']:
        opts.lr = 0.0005
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.n_layer = 3
        opts.dropout = 0.02
        opts.n_batch = 30
        opts.n_tbatch = 30
        opts.use_full_pna = True
        opts.PNA_delta = None
        opts.Gumbel_tau = 1.1
        opts.K = 80
        opts.item_bonus = 0.05
        opts.K_neg = 20
    else:
        raise NotImplemented("No hyper-parameters for this dataset!")
    
    loader = DataLoader(args.data_path, device='cuda' if torch.cuda.is_available() else 'cpu', 
                        K_neg=opts.K_neg)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.n_users = loader.n_users   
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes

    # config_str = '%d,%.6f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.K,opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    config_str = f'K: {opts.K}, lr: {opts.lr}, decay_rate: {opts.decay_rate}, lamb: {opts.lamb}, hidden_dim: {opts.hidden_dim}, n_layer: {opts.n_layer}, n_batch: {opts.n_batch}, dropout: {opts.dropout}\n'
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader)

    # Initialize logger
    logger = SubgraphLogger(results_dir=results_dir, dataset_name=dataset)

    best_recall = 0
    best_summary = None

    for epoch in range(15):
    
        print('epoch ',epoch)
        recall,ndcg, out_str = model.train_batch(logger)
        
        # --- LOGGING ---
        # Compute epoch summary
        summary = logger.compute_epoch_summary(
            n_params=model.n_params,
            train_time=model.t_time,
            inference_time=model.i_time,
            recall=recall,
            ndcg=ndcg
        )
        
        # Save epoch log
        log_file = logger.save_epoch_log(epoch, summary)
        print(f"[INFO] Saved epoch {epoch} log to {log_file}")
        
        # Print formatted summary
        print(logger.format_summary_string(summary))
        
        # Save to text file
        with open(opts.perf_file, 'a+') as f:
            f.write(f"Epoch {epoch}\n")
            f.write(logger.format_summary_string(summary))
            f.write("\n")
        # ------------------
        
        with open(opts.perf_file, 'a+') as f:
            f.write(str(epoch) + out_str)

        if recall > best_recall:
            best_recall = recall
            best_str = out_str
            best_summary = summary
            print("[BEST]" + str(epoch) + '\t' + best_str)
            
    # Save best model summary
    if best_summary is not None:
        best_log_file = logger.save_best_model_log(best_summary)
        print(f"[INFO] Saved best model summary to {best_log_file}")
    
    with open(opts.perf_file, 'a+') as f:
        f.write('best:\n' + best_str)

    print(best_str)
    print(f"\n[INFO] All logs saved to {logger.logs_dir}/")
