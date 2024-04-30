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

def mlp_bi_class(features,labels):
    tuple_ls = list(zip(features, labels))
    #########
    # mlp
    #########
    batchsize = 64
    drop_last = True
    n_feats = 2048
    n_hiddens = 256
    n_tasks = 2

    model = MLP(n_feats=n_feats, n_hiddens=n_hiddens, n_tasks=n_tasks)
    kfolds = BaseDataLoader.load_data_kfold_torch_batchsize(tuple_ls, batchsize=batchsize, Stratify=True, drop_last=drop_last, sample_method=None)
    print('mlp start training!')
    rst_mlp,epoch_mlp = train_mlp.train_bi_classify_kfolds(model, kfolds=kfolds, max_epochs=500, lr=0.0001, save_folder=parent_dir+'/pretrained/',save_name='mlp_train.pth')
    print('optimization finished!', rst_mlp)
    print('best_epoch:',epoch_mlp)
    return rst_mlp


def test_mlp():

    def morgan_featurizer(smiles, nBits=2048):
        mol = Chem.MolFromSmiles(smiles)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits))

    datasets = ['BACE','BBBP2']
    total_rst = []
    for dataset in datasets:
        print('##############'+dataset)
        data_path = parent_dir+f"/dataset/public/{dataset}.csv"
        df = pd.read_csv(data_path,sep='\t',header=0)
        smileses = df['Smiles']
        labels = df['Label'].astype(int)
        print('read data:', Counter(labels))
        nBits = 2048
        features = np.array([morgan_featurizer(smiles, nBits=nBits) for smiles in smileses])
        total_rst.append(mlp_bi_class(features,labels))
    np.savetxt(parent_dir+"/analysis/model_result/test_mlp", np.array(total_rst))
test_mlp()