import os
import pandas as pd
from data_clean import data_preprocess, statistics
from load_data import *
from feature import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from model import MLP, MPNN
from trainer import train_mlp, train_ml, train_gnn
import numpy as np

parent_dir = os.path.abspath(os.path.dirname(__file__))
print(parent_dir)


data_path = parent_dir+"/dataset/train/cleaned_data.csv"

df = pd.read_csv(data_path,sep='\t',header=0)
smileses = df['Smiles']
labels = df['Label'].astype(int)
print('read data:', Counter(labels))
graphs = [Graph_smiles(smiles) for smiles in smileses]
nBits = 2048
features = np.array([morgan_featurizer(smiles, nBits=nBits) for smiles in smileses])

def mlp_bi_class():
    tuple_ls = list(zip(features, labels))
    #########
    # mlp
    #########
    batchsize = int(len(tuple_ls)/16)
    drop_last = True
    n_feats = 2048
    n_hiddens = 256
    n_tasks = 2

    model = MLP(n_feats=n_feats, n_hiddens=n_hiddens, n_tasks=n_tasks)
    kfolds = BaseDataLoader.load_data_kfold_torch_batchsize(tuple_ls, batchsize=batchsize, Stratify=True, drop_last=drop_last, sample_method=None)
    print('mlp start training!')
    rst_mlp,epoch_mlp = train_mlp.train_bi_classify_kfolds(model, kfolds=kfolds, max_epochs=500, save_folder=parent_dir+'/pretrained/',save_name='mlp_train.pth')
    print('optimization finished!', rst_mlp)
    print('best_epoch:',epoch_mlp)
    return rst_mlp


def svm_bi_class():
    tuple_ls = list(zip(features, labels))
    #########
    # svm
    #########
    kfolds = BaseDataLoader.load_data_kfold_numpy(tuple_ls, Stratify=True)
    from sklearn import svm
    import sklearn.model_selection as ms
    # params = [{'kernel':['linear'], 'C':[0.01, 0.1, 1, 10, 100,]},
    #         {'kernel':['poly'], 'C':[0.01, 0.1, 1, 10, 100,], 'degree':[2, 3]}, 
    #         {'kernel':['rbf'], 'C':[0.01, 0.1, 1,10,100,1000], 'gamma':[1, 0.1, 0.01, 0.001]}]
    # model = ms.GridSearchCV(svm.SVC(probability=True), params, cv=5)
    # model.fit(features, labels)
    # print(model.best_params_)
    # svm_paras = model.best_params_
    svm_paras = {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}
    svm_paras['probability']=True
    model =svm.SVC(**svm_paras)
    print('svm start training!')
    rst_svm = train_ml.train_bi_classify_kfolds(model=model, kfolds=kfolds)
    print('optimization finished!', rst_svm)
    return rst_svm

def rf_bi_class():
    tuple_ls = list(zip(features, labels))
    #########
    # rf
    #########
    kfolds = BaseDataLoader.load_data_kfold_numpy(tuple_ls, Stratify=True)
    from sklearn.ensemble import RandomForestClassifier
    import sklearn.model_selection as ms
    # params = [{'n_estimators':[20, 100, 500, 1000, 1500], 
    #             'min_samples_split':[2, 5, 20, 50], 
    #             "min_samples_leaf": [1, 5, 20, 50],
    #             'max_features':["sqrt", "log2", None],
    #             'max_depth':[3, 6, 12]}]
    # model = ms.GridSearchCV(RandomForestClassifier(), params, cv=5)
    # model.fit(features, labels)
    # print(model.best_params_)
    # rf_paras = model.best_params_
    rf_paras = {'max_depth': 3, 'max_features': None, 'min_samples_leaf': 5, 'min_samples_split': 20, 'n_estimators': 20}
    model = RandomForestClassifier(**rf_paras)
    print('rf start training!')
    rst_rf = train_ml.train_bi_classify_kfolds(model=model, kfolds=kfolds)
    print('optimization finished!', rst_rf)
    return rst_rf

def gnn_bi_class():
    tuple_ls = list(zip(graphs, labels))
    #########
    # svm
    #########
    n_feats = 74
    edge_in_feats = 12
    n_tasks = 2
    node_out_feats = 256
    edge_hidden_feats = 256
    batchsize = int(len(tuple_ls)/16)

    model = MPNN(node_in_feats=n_feats, edge_in_feats=edge_in_feats, node_out_feats=node_out_feats, edge_hidden_feats=edge_hidden_feats, n_tasks=n_tasks)
    kfolds = BaseDataLoader.load_data_kfold_graph_batchsize(tuple_ls, batchsize, Stratify=True, drop_last=True)
    print('mpnn start training!')
    rst_gnn, epoch_gnn = train_gnn.train_bi_classify_kfolds(model, kfolds=kfolds, edge=True, max_epochs=500, save_path=parent_dir+'/pretrained/gnn_train.pth')
    print('optimization finished!', rst_gnn)

    return rst_gnn

def total():
    total_rst = []
    for i in range(1):
        rst_mlp = mlp_bi_class()
        rst_svm = svm_bi_class()
        rst_rf = rf_bi_class()
        rst_gnn = gnn_bi_class()
        total_rst += [rst_mlp, rst_rf, rst_svm, rst_gnn]
    np.savetxt(parent_dir+"/analysis/model_result/performance.txt", np.array(total_rst))
# total()

def train():
    tuple_ls = list(zip(features, labels))
    batchsize = int(len(tuple_ls)/16)
    n_feats = 2048
    n_hiddens = 256
    n_tasks = 2
    model = MLP(n_feats=n_feats, n_hiddens=n_hiddens, n_tasks=n_tasks)
    all = BaseDataLoader.load_data_all_torch_batchsize(tuple_ls, batchsize, drop_last=True)
    rst_mlp = train_mlp.train_bi_classify_all(model, all=all, epochs=8, save_folder=parent_dir+'/pretrained/',save_name='all_mlp.pth')
train()