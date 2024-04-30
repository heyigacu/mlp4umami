
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import random
import dgl

class BaseDataLoader(object):
    @staticmethod
    def dgl_collate(sample):
        graphs, labels = map(list,zip(*sample))
        batched_graph = dgl.batch(graphs)
        batched_graph.set_n_initializer(dgl.init.zero_initializer)
        batched_graph.set_e_initializer(dgl.init.zero_initializer)
        return batched_graph, torch.tensor(labels)

    @staticmethod
    def load_data_kfold_numpy(tuple_ls, Stratify=True):
        """
        args:
            ls: [(feature,label)]
            batchsize: int
        """
        random.shuffle(tuple_ls)
        features, labels = list(zip(*tuple_ls))
        if Stratify:
            kf = StratifiedKFold(n_splits=5,shuffle=True)
        else:
            kf = KFold(n_splits=5,shuffle=True)
        kfolds=[]
        for train_idxs,val_idxs in kf.split(features, labels):
            trains = [tuple_ls[index] for index in train_idxs]
            vals = [tuple_ls[index] for index in val_idxs]
            kfolds.append((trains,vals))
        return kfolds
    
    @staticmethod
    def load_data_all_torch_batchsize(tuple_ls, batchsize, drop_last=True):
        random.shuffle(tuple_ls)
        return DataLoader(tuple_ls, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=drop_last)
    
    @staticmethod
    def load_data_kfold_torch_batchsize(tuple_ls, batchsize, Stratify=True, drop_last=True, sample_method=None):
        random.shuffle(tuple_ls)
        features, labels = tuple(zip(*tuple_ls))
        if Stratify:
            kf = StratifiedKFold(n_splits=5,shuffle=True)
        else:
            kf = KFold(n_splits=5,shuffle=True)
        kfolds=[]
        for train_idxs,val_idxs in kf.split(features, labels):
            trains = [tuple_ls[index] for index in train_idxs]
            if sample_method != None:
                (trains_x, trains_y) = tuple(zip(*trains))
                trains_x_sampled, trains_y_sampled = sample_method(np.array(trains_x), list(trains_y))
                trains_x_sampled = [trains_x_sampled[i] for i in range(trains_x_sampled.shape[0])]
                trains = list(zip(trains_x_sampled, trains_y_sampled))
            trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=drop_last)
            vals = [tuple_ls[index] for index in val_idxs]
            vals = DataLoader(vals,batch_size=len(vals), shuffle=True,)
            kfolds.append((trains,vals))
        return kfolds
    
    @staticmethod
    def load_data_all_graph_batchsize(tuple_ls, batchsize, drop_last=True):
        random.shuffle(tuple_ls)
        return DataLoader(tuple_ls, batch_size=batchsize, shuffle=True, collate_fn=BaseDataLoader.dgl_collate, drop_last=drop_last)
    
    @staticmethod
    def load_data_kfold_graph_batchsize(tuple_ls, batchsize, Stratify=True, drop_last=True):
        random.shuffle(tuple_ls)
        features, labels = list(zip(*tuple_ls))
        if Stratify:
            kf = StratifiedKFold(n_splits=5,shuffle=True)
            kf_split = kf.split(features, labels)
        else:
            kf = KFold(n_splits=5,shuffle=True)
            kf_split = kf.split(labels)
        kfolds=[]
        for train_idxs,val_idxs in kf_split:
            trains = [tuple_ls[index] for index in train_idxs]
            trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=BaseDataLoader.dgl_collate, drop_last=drop_last)
            vals = [tuple_ls[index] for index in val_idxs]
            vals = DataLoader(vals,batch_size=len(vals), shuffle=True, collate_fn=BaseDataLoader.dgl_collate, drop_last=drop_last)
            kfolds.append((trains,vals))
        return kfolds




