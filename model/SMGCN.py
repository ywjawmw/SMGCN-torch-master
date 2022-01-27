# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 10:18
# @Author  : Ywj
# @File    : SMGCN.py
# @Description :  the pytorch version of SMGCN

import os
import sys
from utils.helper import *
from utils.batch_test import *
import datetime
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class SMGCN(nn.Module):
    def __init__(self, data_config, pretrain_data):
        super(SMGCN, self).__init__()
        self.model_type = 'SMGCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']
        self.sym_pair_adj = data_config['sym_pair_adj']
        self.herb_pair_adj = data_config['herb_pair_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        # self.link_lr = args.link_lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.loss_weight = args.loss_weight

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.device = args.device


        self.fusion = args.fusion
        print('***********fusion method************ ', self.fusion)

        self.mlp_predict_weight_size = eval(args.mlp_layer_size)
        self.mlp_predict_n_layers = len(self.mlp_predict_weight_size)
        print('mlp predict weight ', self.mlp_predict_weight_size)
        print('mlp_predict layer ', self.mlp_predict_n_layers)
        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        print('regs ', self.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        '''
        *********************************************************
        Create embedding for Input Data & Dropout.
        '''
        # placeholder definition

        # self.users = tf.placeholder(tf.float32, shape=(None, self.n_users))
        # self.user_set = tf.placeholder(tf.int32, shape=(None,))
        #
        # self.items = tf.placeholder(tf.float32, shape=(None, self.n_items))
        # self.item_set = tf.placeholder(tf.int32, shape=(None,))
        #
        # self.pos_items = tf.placeholder(tf.int32, shape=(None,))

        self.mess_dropout = args.mess_dropout

        # self.item_weights = tf.placeholder(tf.float32, shape=(self.n_items, 1))


        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights)
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Build link prediction model
        """
        # self._build_model_phase_II()

        """
        *********************************************************
        Compute link prediction loss
        """
        # self._build_loss_phase_II()


    # 初始化权重，存在all weight字典中，键为权重的名字，值为权重的值
    def _init_weights(self):
        # xavier init
        initializer = nn.init.xavier_uniform_
        all_weights = nn.ParameterDict()
        all_weights.update({'user_embedding': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim)))})
        all_weights.update({'item_embedding': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim)))})
        # all_weights['user_embedding'] = nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim)))
        # all_weights['item_embedding'] = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim)))
        if self.pretrain_data is None:
            print('using xavier initialization')
        else:
            # pretrain
            all_weights['user_embedding'].data = self.pretrain_data['user_embed']
            all_weights['item_embedding'].data = self.pretrain_data['item_embed']
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size    # [embedding size, layer_size]
        pair_dimension = self.weight_size_list[len(self.weight_size_list) - 1]
        for k in range(self.n_layers):
            w_gc_user = torch.empty([2 * self.weight_size_list[k], self.weight_size_list[k + 1]])
            b_gc_user = torch.empty([1, self.weight_size_list[k + 1]])
            W_gc_item = torch.empty([2 * self.weight_size_list[k], self.weight_size_list[k + 1]])
            b_gc_item = torch.empty([1, self.weight_size_list[k + 1]])
            Q_user = torch.empty([self.weight_size_list[k], self.weight_size_list[k]])
            Q_item = torch.empty([self.weight_size_list[k], self.weight_size_list[k]])
            all_weights.update({'W_gc_user_%d' % k: nn.Parameter(initializer(w_gc_user))})
            all_weights.update({'b_gc_user_%d' % k: nn.Parameter(initializer(b_gc_user))})
            all_weights.update({'W_gc_item_%d' % k: nn.Parameter(initializer(W_gc_item))})
            all_weights.update({'b_gc_item_%d' % k: nn.Parameter(initializer(b_gc_item))})
            all_weights.update({'Q_user_%d' % k: nn.Parameter(initializer(Q_user))})
            all_weights.update({'Q_item_%d' % k: nn.Parameter(initializer(Q_item))})
            # all_weights['W_gc_user_%d' % k] = nn.Parameter(initializer(w_gc_user))
            # all_weights['b_gc_user_%d' % k] = nn.Parameter(initializer(b_gc_user))
            # all_weights['W_gc_item_%d' % k] = nn.Parameter(initializer(W_gc_item))
            # all_weights['b_gc_item_%d' % k] = nn.Parameter(initializer(b_gc_item))
            # all_weights['Q_user_%d' % k] = nn.Parameter(initializer(Q_user))
            # all_weights['Q_item_%d' % k] = nn.Parameter(initializer(Q_item))

        self.mlp_predict_weight_size_list = [self.mlp_predict_weight_size[
                                                 len(self.mlp_predict_weight_size) - 1]] + self.mlp_predict_weight_size
        print('mlp_predict_weight_size_list ', self.mlp_predict_weight_size_list)
        for k in range(self.mlp_predict_n_layers):
            W_predict_mlp_user = torch.empty([self.mlp_predict_weight_size_list[k], self.mlp_predict_weight_size_list[k + 1]])
            b_predict_mlp_user = torch.empty([1, self.mlp_predict_weight_size_list[k + 1]])
            all_weights.update({'W_predict_mlp_user_%d' % k: nn.Parameter(initializer(W_predict_mlp_user))})
            all_weights.update({'b_predict_mlp_user_%d' % k: nn.Parameter(initializer(b_predict_mlp_user))})
            # all_weights['W_predict_mlp_user_%d' % k] = nn.Parameter(initializer(W_predict_mlp_user))
            # all_weights['b_predict_mlp_user_%d' % k] = nn.Parameter(initializer(b_predict_mlp_user))
        print("\n", "#" * 75, "pair_dimension is ", pair_dimension)
        M_user = torch.empty([self.emb_dim, pair_dimension])
        M_item = torch.empty([self.emb_dim, pair_dimension])
        all_weights.update({'M_user': nn.Parameter(initializer(M_user))})
        all_weights.update({'M_item': nn.Parameter(initializer(M_item))})
        # all_weights['M_user'] = nn.Parameter(initializer(M_user))
        # all_weights['M_item'] = nn.Parameter(initializer(M_item))
        return all_weights

    # todo: debug check function
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.tensor([coo.row, coo.col], dtype=torch.long).to(args.device)
        v = torch.from_numpy(coo.data).float().to(args.device)
        # coo = X.tocoo().astype(np.float32)
        # indices = np.mat([coo.row, coo.col]).transpose()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    # todo: debug check function
    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]).to(self.device))
        return A_fold_hat

    # 使用图卷积神经网络得到的user embedding
    def _create_graphsage_user_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)    # list
        pre_embeddings = torch.cat([self.weights['user_embedding'], self.weights['item_embedding']], 0)

        # print("*" * 20, "embeddings", pre_embeddings)
        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], pre_embeddings))
            embeddings = torch.cat(temp_embed, 0)
            embeddings = torch.tanh(torch.matmul(embeddings, self.weights['Q_user_%d' % k]))
            embeddings = torch.cat([pre_embeddings, embeddings], 1)

            pre_embeddings = torch.tanh(
                torch.matmul(embeddings, self.weights['W_gc_user_%d' % k]) + self.weights['b_gc_user_%d' % k])
            pre_embeddings = nn.Dropout(self.mess_dropout[k])(pre_embeddings)

            norm_embeddings = F.normalize(pre_embeddings, p=2, dim=1)
            all_embeddings = [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        temp = torch.sparse.mm(self._convert_sp_mat_to_sp_tensor(self.sym_pair_adj).to(self.device),
                                             self.weights['user_embedding'])
        user_pair_embeddings = torch.tanh(torch.matmul(temp, self.weights['M_user']))

        if self.fusion in ['add']:
            u_g_embeddings = u_g_embeddings + user_pair_embeddings
        if self.fusion in ['concat']:
            u_g_embeddings = torch.cat([u_g_embeddings, user_pair_embeddings], 1)
        return u_g_embeddings

    def _create_graphsage_item_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        pre_embeddings = torch.cat([self.weights['user_embedding'], self.weights['item_embedding']], 0)

        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], pre_embeddings))
            embeddings = torch.cat(temp_embed, 0)

            embeddings = torch.tanh(torch.matmul(embeddings, self.weights['Q_item_%d' % k]))
            embeddings = torch.cat([pre_embeddings, embeddings], 1)

            pre_embeddings = torch.tanh(
                torch.matmul(embeddings, self.weights['W_gc_item_%d' % k]) + self.weights['b_gc_item_%d' % k])

            pre_embeddings = nn.Dropout(self.mess_dropout[k])(pre_embeddings)

            norm_embeddings = F.normalize(pre_embeddings, p=2, dim=1)
            all_embeddings = [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        temp = torch.sparse.mm(self._convert_sp_mat_to_sp_tensor(self.herb_pair_adj).to(self.device),
                                             self.weights['item_embedding'])
        item_pair_embeddings = torch.tanh(torch.matmul(temp, self.weights['M_item']))

        if self.fusion in ['add']:
            i_g_embeddings = i_g_embeddings + item_pair_embeddings

        if self.fusion in ['concat']:
            i_g_embeddings = torch.cat([i_g_embeddings, item_pair_embeddings], 1)

        return i_g_embeddings

    def create_batch_rating(self, pos_items, user_embeddings):
        # sum_embeddings = torch.matmul(users, ua_embeddings)
        #
        # normal_matrix = torch.reciprocal(torch.sum(users, 1))
        #
        # normal_matrix = normal_matrix.unsqueeze(1)
        #
        # extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])
        #
        # user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)
        #
        # for k in range(0, self.mlp_predict_n_layers):
        #     user_embeddings = F.relu(
        #         torch.matmul(user_embeddings,
        #                      self.weights['W_predict_mlp_user_%d' % k]) + self.weights['b_predict_mlp_user_%d' % k])
        #     user_embeddings = F.dropout(user_embeddings, 1 - self.mess_dropout[k])

        pos_scores = torch.sigmoid(torch.matmul(user_embeddings, pos_items.transpose(0, 1)))
        return pos_scores

    def create_set2set_loss(self, items, item_weights, user_embeddings, all_user_embeddins, ia_embeddings):
        # sum_embeddings = torch.matmul(users, ua_embeddings)   # [B, embedding_size]
        #
        # normal_matrix = torch.reciprocal(torch.sum(users, 1))
        #
        # normal_matrix = normal_matrix.unsqueeze(1)      # [B, 1]
        # # 复制embedding_size列  [B, embedding_size]
        # extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])
        # # 对应元素相乘
        # user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)
        # # print("*" * 20, "user_embeddings", user_embeddings[0][20:30])
        # all_user_embeddins = torch.index_select(ua_embeddings, 0, user_set)  # []
        #
        # for k in range(0, self.mlp_predict_n_layers):
        #     user_embeddings = F.relu(
        #         torch.matmul(user_embeddings, self.weights['W_predict_mlp_user_%d' % k])
        #         + self.weights['b_predict_mlp_user_%d' % k])
        #
        #     user_embeddings = F.dropout(user_embeddings, 1 - self.mess_dropout[k])
            # print("*" * 20, "user_embeddings", user_embeddings)
        # print("*" * 20, "user_embeddings", torch.unique(user_embeddings))

        predict_probs = torch.sigmoid(torch.matmul(user_embeddings, ia_embeddings.transpose(0, 1)))
        # print("*" * 20, "items", items[0]-predict_probs[0])
        # print(item_weights)
        mf_loss = torch.sum(torch.matmul(torch.square((items - predict_probs)), item_weights), 0)
        # mf_loss = nn.MSELoss(reduction='elementwise_mean')(items, predict_probs)
        mf_loss = mf_loss / self.batch_size

        all_item_embeddins = ia_embeddings
        regularizer = torch.norm(all_user_embeddins) ** 2 / 2 + torch.norm(all_item_embeddins) ** 2 / 2
        regularizer = regularizer.reshape(1)
        # F.normalize(all_user_embeddins, p=2) + F.normalize(all_item_embeddins, p=2)
        regularizer = regularizer / self.batch_size

        emb_loss = self.decay * regularizer

        reg_loss = torch.tensor([0.0], dtype=torch.float64, requires_grad=True).to(self.device)
            # torch.nn.init.constant(reg_loss, 0.0)
        # loss = mf_loss + emb_loss + reg_loss

        return mf_loss, emb_loss, reg_loss

    def forward(self, users, user_set=None, pos_items=None, train=True):
        """
          *********************************************************
          Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
          Different Convolutional Layers:
              1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
              2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
              3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
          """
        # todo: todo 应该在主函数中
        if self.alg_type in ['SMGCN']:
            if train:
                ua_embeddings = self._create_graphsage_user_embed()
                ia_embeddings = self._create_graphsage_item_embed()

                sum_embeddings = torch.matmul(users, ua_embeddings)  # [B, embedding_size]
                normal_matrix = torch.reciprocal(torch.sum(users, 1))
                normal_matrix = normal_matrix.unsqueeze(1)  # [B, 1]
                # 复制embedding_size列  [B, embedding_size]
                extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])
                # 对应元素相乘
                user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)
                # print("*" * 20, "user_embeddings", user_embeddings[0][20:30])
                all_user_embeddins = torch.index_select(ua_embeddings, 0, user_set)  # []
                # print("*" * 20, "all_user_embeddings", all_user_embeddins[0][20:30])

                for k in range(0, self.mlp_predict_n_layers):
                    user_embeddings = F.relu(
                        torch.matmul(user_embeddings, self.weights['W_predict_mlp_user_%d' % k])
                        + self.weights['b_predict_mlp_user_%d' % k])
                    user_embeddings = F.dropout(user_embeddings, self.mess_dropout[k])
                # print("*" * 20, "user_embeddings", user_embeddings[0])
                return user_embeddings, all_user_embeddins, ia_embeddings
            else:
                ua_embeddings = self._create_graphsage_user_embed()
                ia_embeddings = self._create_graphsage_item_embed()
                pos_items = torch.tensor(pos_items, dtype=torch.long).to(args.device)
                pos_i_g_embeddings = torch.index_select(ia_embeddings, 0, pos_items)
                sum_embeddings = torch.matmul(users, ua_embeddings)

                normal_matrix = torch.reciprocal(torch.sum(users, 1))

                normal_matrix = normal_matrix.unsqueeze(1)

                extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])

                user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)

                for k in range(0, self.mlp_predict_n_layers):
                    user_embeddings = F.relu(
                        torch.matmul(user_embeddings,
                                     self.weights['W_predict_mlp_user_%d' % k]) + self.weights[
                            'b_predict_mlp_user_%d' % k])
                    user_embeddings = F.dropout(user_embeddings, self.mess_dropout[k])
                return user_embeddings, pos_i_g_embeddings
            # else:
            #     pos_items = torch.tensor(pos_items, dtype=torch.long).to(args.device)
            #     pos_i_g_embeddings = torch.index_select(ia_embeddings, 0, pos_items)
            #
            #     batch_ratings = self.create_batch_rating(users, self.pos_i_g_embeddings)
            #     return batch_ratings











