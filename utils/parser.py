# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 10:40
# @Author  : Ywj
# @File    : parser.py
# @Description :  SMGCN参数构造
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run SMGCN.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--result_index', type=int, default=1,
                        help='result file index.')
    parser.add_argument('--test_file', nargs='?', default='valid_id.txt',
                        help='valid_5percent.txt')

    parser.add_argument('--result_label', nargs='?', default='',
                        help='result path label.')

    parser.add_argument('--is_test_data', type=int, default=0,
                        help='flag for test data.')


    parser.add_argument('--dataset', nargs='?', default='Herb',
                        help='Choose a dataset from {Herb, NetEase, gowalla, yelp2018, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')

    parser.add_argument('--modeltype', nargs='?', default='ngcf',
                        help='ngcf or gcnversion.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[128,256]',
                        help='Output sizes of every layer')
    parser.add_argument('--pair_layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')
    parser.add_argument('--mlp_layer_size', nargs='?', default='[256]',
                        help='Output sizes of every layer')
    parser.add_argument('--attention_size', type=int, default=64,
                        help='attention W size.')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')



    parser.add_argument('--fusion', nargs='?', default='add',
                        help='fusion method.')
    parser.add_argument('--neg_sample', type=int, default=1,
                        help='neg sample num.')

    parser.add_argument('--regs', nargs='?', default='[7e-3]',
                        help='embed Regularizations.')
    parser.add_argument('--link_regs', nargs='?', default='[0]',
                        help='link embed Regularizations.')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate.')
    parser.add_argument('--link_lr', type=float, default=0.01,
                        help='Learning rate for link prediction.')
    parser.add_argument('--k_clf', type=float, default=0.95,
                        help='Link rate for k clique.')
    parser.add_argument('--top_core', type=int, default=0,
                        help='0 for not using top core as the clique core, 1 for else')

    parser.add_argument('--model_type', nargs='?', default='ngcf',
                        help='Specify the name of model (ngcf).')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='SMGCN',
                        help='Specify the type of the graph convolutional layer from {SMGCN, ngcf, gcn, gcmc}.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.0,0.0]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout_link', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')


    parser.add_argument('--Ks', nargs='?', default='[5,10,15,20]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--loss_weight', type=float, default=1.0,
                        help='number:0-1 change different loss')


    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_args()
