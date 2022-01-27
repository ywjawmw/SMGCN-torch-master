# -*- coding: utf-8 -*-
# @Time    : 2022/1/17 16:22
# @Author  : Ywj
# @File    : smgcn_main.py
# @Description :  SMGCN主函数

import numpy as np
import os
import sys
from model.SMGCN import SMGCN
import datetime
from utils.helper import *
import torch
from utils.batch_test import *
import torch.optim as optim


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    startTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start ', startTime)
    print('************SMGCN*************** ')
    print('result_index ', args.result_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.device = torch.device('cuda:' + str(args.gpu_id))

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, sym_pair_adj, herb_pair_adj = data_generator.get_adj_mat()
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)


    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj


    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj


    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj

    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = SMGCN(data_config=config, pretrain_data=pretrain_data).to(args.device)
    print(model)

    """
    *********************************************************
    Save the model parameters.
    """

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        mess_dr = '-'.join([str(l) for l in eval(args.mess_dropout)])
        weights_save_path = '%sweights-SMGCN/%s/%s/%s/%s/l%s_r%s_messdr%s/' % (
        args.weights_path, args.dataset, model.model_type, layer, args.embed_size,
        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), mess_dr)
        ensureDir(weights_save_path)
        # print("\n", "*" * 80, "model sava path", weights_save_path + 'model.pkl')
        # torch.save(model, weights_save_path + 'model.pkl')

    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    print("args.pretrain\t", args.pretrain)
    if args.pretrain == 1:
        print("pretrain==1")
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        mess_dr = '-'.join([str(l) for l in eval(args.mess_dropout)])
        weights_save_path = '%sweights-SMGCN/%s/%s/%s/%s/l%s_r%s_messdr%s/' % (
            args.weights_path, args.dataset, model.model_type, layer, args.embed_size,
            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), mess_dr)
        # weights_save_path = weights_save_path.replace('[', '_').replace(']', '_')
        pretrain_path = weights_save_path
        print('load the pretrained model parameters from: ', pretrain_path)
        model = torch.load(weights_save_path + 'model.pkl')
        if model:
            print("start to load pretrained model")
            if args.report != 1:
                ret = test(model, list(data_generator.test_users),
                           data_generator.test_group_set, drop_flag=True)
                cur_best_pre_0 = ret['precision'][0]
                pretrain_ret = 'pretrained model recall=%.5f, precision=%.5f, ' \
                                'ndcg=%.5f, RMRR=%.5f' % \
                               (ret['recall'][0], ret['precision'][0], ret['ndcg'][0], ret['rmrr'][0])
                print(pretrain_ret)
        else:
            cur_best_pre_0 = 0.
            print('no model, without pretraining.')

    else:
        cur_best_pre_0 = 0.
        print('no pretrain, without pretraining.')

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, rmrr_loger = [], [], [], [], []
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.

        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            optimizer.zero_grad()
            users, user_set, items, item_set = data_generator.sample()
            # for i in range(3):
            #     # print(users[i])
            #     for j in range(len(users[i])):
            #         if users[i][j] > 0:
            #             print(i, "\t", j)
            users = torch.tensor(users, dtype=torch.float32).to(args.device)
            user_set = torch.tensor(user_set, dtype=torch.long).to(args.device)
            items = torch.tensor(items, dtype=torch.float32).to(args.device)
            item_weights = torch.tensor(data_generator.item_weights, dtype=torch.float32).to(args.device)

            # item_set = torch.from_numpy(item_set).to(args.device)
            user_embeddings, all_user_embeddins, ia_embeddings = model(users, user_set)
            # print("*" * 20, "ua_embeddings", torch.unique(ua_embeddings))
            batch_mf_loss, batch_emb_loss, batch_reg_loss = \
                model.create_set2set_loss( items, item_weights, user_embeddings, all_user_embeddins, ia_embeddings)
            batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss
            # perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  ]' % (
            #     epoch, time() - t1, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss)
            # print(perf_str)
            batch_loss.backward()
            # print(model.weights['W_predict_mlp_user_0'].grad[0])
            optimizer.step()
            # print("&" * 20, ua_embeddings[0][:10])
            # print("& ia" * 20, ia_embeddings[0][:10])
            loss += batch_loss.item()
            mf_loss += batch_mf_loss.item()
            emb_loss += batch_emb_loss.item()
            reg_loss += batch_reg_loss.item()
        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()
        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  ]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss )
                print(perf_str)
            continue

        t2 = time()

        group_to_test = data_generator.test_group_set
        ret = test(model, list(data_generator.test_users), group_to_test, drop_flag=True)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        rmrr_loger.append(ret['rmrr'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f ]\n recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f],  ndcg=[%.5f, %.5f], RMRR=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss,  ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1], ret['rmrr'][0], ret['rmrr'][-1])
            print(perf_str)
            paras = str(args.lr) + "_" + str(args.regs) + "_" + str(args.mess_dropout) + "_" + str(args.embed_size) + "_" + str(
                args.adj_type) + "_" + str(args.alg_type)
            print("paras\t", paras)


        # cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['precision'][0], cur_best_pre_0,
        #                                                             stopping_step, expected_order='acc', flag_step=10)

        cur_best_pre_0, stopping_step, should_stop = no_early_stopping(ret['precision'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc')

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.

        if should_stop == True:
            print('early stopping')
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['precision'][0] == cur_best_pre_0 and args.save_flag == 1:
            print("\n", "*" * 80, "model sava path", weights_save_path + 'model.pkl')
            torch.save(model, weights_save_path + 'model.pkl')
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    rmrrs = np.array(rmrr_loger)

    best_pres_0 = max(pres[:, 0])
    idx = list(pres[:, 0]).index(best_pres_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s],  ndcg=[%s], rmrr=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
                  '\t'.join(['%.5f' % r for r in rmrrs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result-SMGCN-%d' % (args.proj_path, args.dataset, model.model_type, args.result_index)
    ensureDir(save_path)
    f = open(save_path, 'a')

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f.write(
        'time=%s, fusion=%s, embed_size=%d, lr=%s, layer_size=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\t'
        % (str(cur_time), args.fusion, args.embed_size,   str(args.lr), args.layer_size,
           args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()

    endTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('end ', endTime)
