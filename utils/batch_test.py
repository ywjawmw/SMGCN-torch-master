# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 11:10
# @Author  : Ywj
# @File    : batch_test.py
# @Description : 测试核心代码
import numpy as np

from utils.parser import parse_args
from utils.load_data import *
import multiprocessing
import torch
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
# import sklearn.metrics as m

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(args=args, path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


def test(model, users_to_test, test_group_list, drop_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)), 'rmrr': np.zeros(len(Ks))}
    # test_users = users_to_test
    test_users = torch.tensor(users_to_test, dtype=torch.float32).to(args.device)
    item_batch = range(ITEM_NUM)
    if drop_flag == False:
        user_embeddings, pos_i_g_embeddings = model(test_users, pos_items=item_batch, train=False)
    else:
        args.mess_dropout = str([0.] * len(eval(args.layer_size)))
        args.mess_dropout = eval(args.mess_dropout)
        user_embeddings, pos_i_g_embeddings = model(test_users, pos_items=item_batch, train=False)
        print('drop_flag: ', drop_flag, ',\t mess_dropout: ', args.mess_dropout)
    rate_batch = model.create_batch_rating(pos_i_g_embeddings, user_embeddings)
    print('rate_batch ', rate_batch.shape)

    user_batch_rating_uid = zip(test_users, rate_batch)
    user_rating_dict = {}

    index = 0
    for entry in user_batch_rating_uid:
        rating = entry[1]         # (1, 753)
        # print("@@@@@@@@@@@@rating@@@@@@@@")
        # print(rating)
        temp = [(i, float(rating[i])) for i in range(len(rating))]
        user_rating_dict[index] = temp
        index += 1
    # user_rating_dict {sym-1: [(herb1, rate), (herb2, rate), ..., (herb753, rate)], ...,
    #                   sym-1162:[(herb1, rate), (herb2, rate), ..., (herb753, rate)]}

    precision_n = np.zeros(len(Ks))
    recall_n = np.zeros(len(Ks))
    ndcg_n = np.zeros(len(Ks))
    rmrr_n = np.zeros(len(Ks))
    topN = Ks

    gt_count = 0
    candidate_count = 0
    for index in range(len(test_group_list)):
        entry = test_group_list[index]
        v = entry[1]                              # sym-index's true herb list
        rating = user_rating_dict[index]

        candidate_count += len(rating)
        rating.sort(key=lambda x: x[1], reverse=True)
        gt_count += len(v)
        K_max = topN[len(topN) - 1]
        r = []
        # number = 0
        # herb_results = []  # 推荐列表中herb 集合
        for i in rating[:K_max]:
            herb = i[0]
            if herb in v:
                r.append(1)
            else:
                r.append(0)

        for ii in range(len(topN)):      # topN: [5, 10, 15, 20]
            number = 0
            herb_results = []  # 推荐列表中herb 集合
            for i in rating[:topN[ii]]:
                herb = i[0]
                herb_results.append(herb)
                if herb in v:
                    number += 1
            # todo: modified MRR to Rank-MRR
            mrr_score = 0.
            # print("-----herb_results:", herb_results)
            # print("-----ground truth:", v)
            for a_rank in range(len(v)):  # herb 在grand truth中的位置a_rank
                if v[a_rank] in herb_results:
                    a_refer = herb_results.index(v[a_rank])  # herb 在推荐列表中的位置a_refer
                    mrr_score += 1.0 / (abs(a_refer - a_rank) + 1)
            precision_n[ii] = precision_n[ii] + float(number / topN[ii])
            recall_n[ii] = recall_n[ii] + float(number / len(v))
            ndcg_n[ii] = ndcg_n[ii] + ndcg_at_k(r, topN[ii])
            rmrr_n[ii] = rmrr_n[ii] + mrr_score / len(v)
    print('gt_count ', gt_count)
    print('candidate_count ', candidate_count)
    print('ideal candidate count ', len(test_group_list) * ITEM_NUM)
    for ii in range(len(topN)):
        result['precision'][ii] = precision_n[ii] / len(test_group_list)
        result['recall'][ii] = recall_n[ii] / len(test_group_list)
        result['ndcg'][ii] = ndcg_n[ii] / len(test_group_list)
        result['rmrr'][ii] = rmrr_n[ii] / len(test_group_list)
    return result


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(np.ones_like(r), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


