import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel

parser = argparse.ArgumentParser(description="Parser for Adaptive Subgraph Model")
parser.add_argument('--data_path', type=str, default='data/last-fm/')
parser.add_argument('--seed', type=str, default=42)
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

    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    torch.cuda.set_device(args.gpu)

    loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.n_users = loader.n_users   
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes

    if dataset in ['last-fm-subset']:
        opts.lr = 0.0004
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 30
        opts.n_tbatch = 30
        opts.pruner_hidden_dim = 48 // 2
        opts.use_full_pna = True
        opts.PNA_delta = None
        opts.Gumbel_tau = 1.1
        opts.Gumbel_threshold = 0.3
    else:
        raise NotImplemented("No hyper-parameters for this dataset!")

    config_str = '%d,%.6f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.K,opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader)

    best_recall = 0
    for epoch in range(15):
    
        print('epoch ',epoch)
        recall,ndcg, out_str = model.train_batch()
        
        with open(opts.perf_file, 'a+') as f:
            f.write(str(epoch) + out_str)

        if recall > best_recall:
            best_recall = recall
            best_str = out_str
            print(str(epoch) + '\t' + best_str)
    with open(opts.perf_file, 'a+') as f:
        f.write('best:\n'+best_str)

    print(best_str)

