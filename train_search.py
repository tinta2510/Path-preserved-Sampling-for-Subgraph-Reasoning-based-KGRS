import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel

parser = argparse.ArgumentParser(description="Parser for Adaptive Subgraph Model")
parser.add_argument('--data_path', type=str, default='data/last-fm/')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=int, default=0)


# ===== Hyperparameters =====
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--decay_rate', type=float, default=None)
parser.add_argument('--lamb', type=float, default=None)
parser.add_argument('--hidden_dim', type=int, default=None)
parser.add_argument('--n_layer', type=int, default=None)
parser.add_argument('--dropout', type=float, default=None)
parser.add_argument('--act', type=str, default=None)
parser.add_argument('--n_batch', type=int, default=None)
parser.add_argument('--n_tbatch', type=int, default=None)
parser.add_argument('--use_full_pna', action='store_true')
parser.add_argument('--PNA_delta', type=float, default=None)
parser.add_argument('--K', type=int, default=None)
parser.add_argument('--item_bonus', type=float, default=None)

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

    loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.n_users = loader.n_users   
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes

    opts.lr = args.lr if args.lr is not None else 0.0005
    opts.decay_rate = args.decay_rate if args.decay_rate is not None else 0.994
    opts.lamb = args.lamb if args.lamb is not None else 0.00014
    opts.hidden_dim = args.hidden_dim if args.hidden_dim is not None else 64    
    opts.n_layer = args.n_layer if args.n_layer is not None else 3
    opts.dropout = args.dropout if args.dropout is not None else 0.02
    opts.n_batch = args.n_batch if args.n_batch is not None else (30 if opts.n_layer <=3 else 20)
    opts.n_tbatch = args.n_tbatch if args.n_tbatch is not None else (30 if opts.n_layer <=3 else 20)
    opts.use_full_pna = args.use_full_pna if args.use_full_pna else True
    opts.PNA_delta = args.PNA_delta if args.PNA_delta is not None else None
    opts.K = args.K if args.K is not None else 60
    opts.item_bonus = args.item_bonus if args.item_bonus is not None else 0.05
    opts.K_neg = 20  # default number of negative samples
    
    # config_str = '%d,%.6f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.K,opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    config_str = f'''K: {opts.K}, lr: {opts.lr}, decay_rate: {opts.decay_rate}, lamb: {opts.lamb}, hidden_dim: {opts.hidden_dim}, 
                    n_layer: {opts.n_layer}, n_batch: {opts.n_batch}, dropout: {opts.dropout}, item_bonus: {opts.item_bonus}, K_neg: {opts.K_neg}\n'''
                    
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader)

    best_recall = 0
    for epoch in range(2):
    
        print('epoch ',epoch)
        recall,ndcg, out_str = model.train_batch()
        
        with open(opts.perf_file, 'a+') as f:
            f.write(str(epoch) + out_str)

        if recall > best_recall:
            best_recall = recall
            best_str = out_str
            print("[BEST]" + str(epoch) + '\t' + best_str)
    with open(opts.perf_file, 'a+') as f:
        f.write('best:\n'+best_str)

    print(best_str)

