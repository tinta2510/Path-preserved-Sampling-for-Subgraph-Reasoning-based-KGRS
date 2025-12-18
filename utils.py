import numpy as np
import torch

def cal_bpr_loss(n_users, pos, neg, scores):
 
    n = scores.shape[0] # number of users in the batch

    loss = 0
    for i in range(n):
        pos_score = scores[i][pos[i]-n_users] # List of positive items for each userw
        neg_score = scores[i][neg[i]-n_users] # List of sampled negative items for each user
        u_loss = -1*torch.sum(torch.nn.LogSigmoid()(pos_score - neg_score))
        loss += u_loss

    return loss

def cal_bpr_loss_k_neg(n_users, pos, neg_samples, scores, K=5):
    """
    BPR loss with K negative samples for each positive item.
    
    Args:
        n_users: number of users
        pos: [B, num_pos] array of positive items for each user 
        neg_samples: [B, num_pos, K] array of K negative samples for each positive item
        scores: [B, n_items] tensor of item scores for each user in the batch
        K: number of negative samples per positive
    
    Returns:
        loss: scalar BPR loss
    """
    n = scores.shape[0]  # batch size
    
    loss = 0
    for i in range(n):
        # Get positive scores
        pos_indices = pos[i] - n_users  # [num_pos]
        pos_scores = scores[i][pos_indices]  # [num_pos]
        
        # Get negative scores - [K] negative samples
        neg_indices = neg_samples[i] - n_users # [num_pos, K]
        neg_scores = scores[i][neg_indices]  # [num_pos, K]
                
        # Compute pairwise BPR loss: each positive vs each negative
        # pos_scores: [num_pos], neg_scores: [num_pos, K]
        pos_scores_expanded = pos_scores.unsqueeze(-1)  # [num_pos, 1]
        
        # Broadcast: [num_pos, K]
        score_diff = pos_scores_expanded - neg_scores
                
        # BPR loss over all pairs
        u_loss = -torch.sum(torch.nn.LogSigmoid()(score_diff)) / K
        loss += u_loss
    
    return loss

def ndcg_k(r, k, len_pos_test):

    if len_pos_test > k :
        standard = [1.0] * k
    else:
        standard = [1.0]*len_pos_test + [0.0]*(k - len_pos_test)
    dcg_max = dcg_k(standard, k)
    
    return dcg_k(r, k) / dcg_max

def dcg_k(r, k):

    r = np.asarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))



