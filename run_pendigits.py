# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:48:35 2016

Perform experiments with Pendigits

@author: bo
"""
import sys
import gzip 
import pickle as cPickle
import numpy as np
import os
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from multi_layer_km_semi import test_SdC
from cluster_acc import acc

# 重复运行次数
trials = 3

# 数据地址
dataset_all = [
    'dataset/segment.pkl.gz',
    'dataset/optdigits.pkl.gz',
     'dataset/sat.pkl.gz',
     'dataset/pendigits.pkl.gz',
     'dataset/letter-recognition.pkl.gz'
    ]


for i in range(len(dataset_all)):
    # if not i == 3:
    #     continue
    dataset = dataset_all[i]
    # 读取数据
    with gzip.open(dataset, 'rb') as f:
        train_x, train_y = cPickle.load(f,encoding='iso-8859-1')
        train_x = train_x
        print("test =", type(train_x), type(train_y))
        print(train_x.shape, train_y.shape)
        print(train_x[0], train_y[0])
        print("train_y",train_y, train_y.shape)

    unique_train_y = list(set(train_y))
    for i in range(len(train_y)):
        transform_label = unique_train_y.index(train_y[i])
        train_y[i] = transform_label

    K = len(np.unique(train_y))
    print("K =",K)

    # 此为可视化图像
    # tsne=TSNE()
    # X_embedded = tsne.fit_transform(train_x)
    # print(X_embedded, type(X_embedded), len(X_embedded), X_embedded.shape)
    # color = ['r.', '#FFA500.','g.','b.','k.','y.','m.','c.','y.', 'm.', 'r.']
    # colors=['b', 'c', 'y', 'm', 'r','b', 'c', 'y', 'm', 'r']
    # colorss = ['#000000', '#DEB887', '#DC143C','#000000', '#DEB887', '#DC143C','#000000', '#DEB887', '#DC143C']
    # colorsss = ['coral', 'maroon','gold','b', 'r', 'g','k', 'y', 'm','c' ]
    # for i in range(K):
    #     d = X_embedded[train_y == i]
    #     print(len(d))
    #     plt.scatter(d[:,0], d[:,1], color=colorsss[i], marker = '.'  ) #color[i]
    # plt.show()

    # perform KM  运行K-means算法
    km_model = KMeans(n_clusters = K, n_init = 1)
    results_KM = np.zeros((trials, 3))
    for i in range(trials):
        ypred = km_model.fit_predict(train_x)
        nmi = metrics.adjusted_mutual_info_score(train_y, ypred)
        ari = metrics.adjusted_rand_score(train_y, ypred)
        ac  = acc(ypred, train_y)
        results_KM[i] = np.array([nmi, ari, ac])

    KM_mean = np.mean(results_KM, axis = 0)
    KM_std  = np.std(results_KM, axis = 0)
    # Perform SC
    # print('SC started...')
    # results_SC = np.zeros((trials, 3))
    # se_model = SpectralEmbedding(n_components=K, affinity='rbf', gamma = 0.1)
    # se_vec = se_model.fit_transform(train_x)
    # for i in range(trials):
    #     ypred = km_model.fit_predict(se_vec)
    #     nmi = metrics.adjusted_mutual_info_score(train_y, ypred)
    #     ari = metrics.adjusted_rand_score(train_y, ypred)
    #     ac  = acc(ypred, train_y)
    #     results_SC[i] = np.array([nmi, ari, ac])
    #
    # SC_mean = np.mean(results_SC, axis = 0)
    # SC_std  = np.std(results_SC, axis = 0)

    # 模型JLCN的参数
    config = {'Init': '',
              'pre_alpha': 1, #  pretrain reconstruction 预训练阶段损失函数中重构损失的权重
              'pre_beta': 1,  # pretrain pairwise  预训练阶段损失函数中成对约束损失的权重
              'finetune_alpha': 1,  # finetune cluster center  聚类阶段损失函数中聚类中心损失的权重
              'finetune_beta': 1,  # finetune pairwise  聚类阶段损失函数中重构损失的权重
              'finetune_gamma': 0.5,  # finetune reconstruction  聚类阶段损失函数中成对约束损失的权重
              'pairwise_adaptive': 1,  # -1:None  0:constant  1:adaptive  成对约束损失中代价敏感函数的设置，为1时是自适应代价敏感函数
              'output_dir': 'Pendigits', # 不使用的参数
              'save_file': 'pen_10.pkl.gz', # 不使用的参数
              'pretraining_epochs': 50, # 50  预训练阶段的批次训练数
              'pretrain_lr': 0.01, # 0.01  预训练学习率
              'mu': 0.9,
              'finetune_lr': 0.02, # 0.002  聚类阶段练学习率
              'training_epochs': 50, # 50  聚类阶段的批次训练数
              'dataset': dataset,
              'batch_size': 100, # 100
              'nClass': K,
              'hidden_dim':  [50, 16, 10]  , # [50, 16, 10]  [100,36,20]  [200, 64, 50]  [60,19,15]  [15,10,6] [10,6,4] [12,4,3]
              'diminishing': False}

    if dataset == dataset_all[0]:
        config['hidden_dim'] = [60,19,15]
    elif dataset == dataset_all[1]:
        config['hidden_dim'] = [200, 64, 50]
    elif dataset == dataset_all[2]:
        config['hidden_dim'] = [100,36,20]
    elif dataset == dataset_all[3] or dataset == dataset_all[4]:
        config['hidden_dim'] = [50, 16, 10]

    print(config)

    results = []
    for i in range(trials):
        res_metrics = test_SdC(**config)
        results.append(res_metrics)

    results_SAEKM = np.zeros((trials, 3))
    results_DCN   = np.zeros((trials, 3))

    N = config['training_epochs']//5
    for i in range(trials):
        results_SAEKM[i] = results[i][0]
        results_DCN[i] = results[i][N]
    SAEKM_mean = np.mean(results_SAEKM, axis = 0)
    SAEKM_std  = np.std(results_SAEKM, axis = 0)
    DCN_mean   = np.mean(results_DCN, axis = 0)
    DCN_std    = np.std(results_DCN, axis = 0)

    # 以下是输出结果，DCN（deep clustering network）为本模型的结果
    print (  ('KM avg. NMI = {0:.3f}, ARI = {1:.3f}, ACC = {2:.3f}'.format(KM_mean[0],
                          KM_mean[1], KM_mean[2]) )  )
    # print (  ('SC   avg. NMI = {0:.3f}, ARI = {1:.3f}, ACC = {2:.3f}'.format(SC_mean[0],
    #                       SC_mean[1], SC_mean[2]) )  )
    print (  ('SAE+KM avg. NMI = {0:.3f}, ARI = {1:.3f}, ACC = {2:.3f}'.format(SAEKM_mean[0],
                          SAEKM_mean[1], SAEKM_mean[2]) )    )
    print (  ('DCN    avg. NMI = {0:.3f}, ARI = {1:.3f}, ACC = {2:.3f}'.format(DCN_mean[0],
                          DCN_mean[1], DCN_mean[2]) )  )

    f = open('/home/lee/PycharmProjects/DCN-master/result/parameter_saved_finetune_no_adaptive_20190127_adaptive=0.1', 'a')
    f.write(dataset)
    f.write("\n")
    f.write("pre_alpha:"+str(config['pre_alpha'])+"  pre_beta:"+str(config['pre_beta'])+
            "  finetune_alpha:"+str(config['finetune_alpha'])+"  finetune_beta:"+str(config['finetune_beta'])
            +"  pairwise_adaptive:"+str(config['pairwise_adaptive']))
    f.write("\n")
    f.write('SAE+KM avg. NMI = {0:.3f}, ARI = {1:.3f}, ACC = {2:.3f}'.format(SAEKM_mean[0],
                          SAEKM_mean[1], SAEKM_mean[2]))
    f.write("\n")
    f.write('DCN    avg. NMI = {0:.3f}, ARI = {1:.3f}, ACC = {2:.3f}'.format(DCN_mean[0],
                          DCN_mean[1], DCN_mean[2]))
    f.write("\n")
    f.write(str(DCN_mean[0]) + " ,"+ str(DCN_mean[1]) +" ,"+ str(DCN_mean[2]))
    f.write("\n\n")
    f.close()
    print(dataset)
