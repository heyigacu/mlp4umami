import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import MPNNGNN, GraphSAGE
from dgl.nn.pytorch import Set2Set
import torch.nn.functional as F
from dgllife.model.readout import WeightedSumAndMax

class MLP(nn.Module):
    def __init__(self, n_feats, n_hiddens, n_tasks):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_feats, n_hiddens),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_hiddens, n_tasks),
        )
        self.criteon = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.model(x.to(torch.float32))
        return x

class MPNN(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=2,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNN, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)





