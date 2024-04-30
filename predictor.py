import argparse
import torch
import os
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import pandas as pd
from model import MLP
from feature import morgan_featurizer

work_dir = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='molecules odor predictor')

parser.add_argument("-i", "--input", type=str, default=work_dir+'/input.csv', help="input file")
parser.add_argument("-o", "--output", type=str, default=work_dir+'/result.csv',help="output file")
args = parser.parse_args()

##################
# predict
##################

sequences = list(pd.read_csv(args.input, header=0, sep=' ')['Sequence'])

def mlp_bi(model, path, sequences):
    state_dict = torch.load(work_dir+"/"+path)
    model.load_state_dict(state_dict)
    model.eval()
    total = []
    for i,sequence in enumerate(sequences):
        try:
            if i % 10000 ==0:
                print(i)
            for i in list(DataLoader([morgan_featurizer(sequence)], batch_size=1, shuffle=False, collate_fn=None, drop_last=False)):
                feature = i
            rst = model(feature)
            rst =  F.softmax(rst,dim=1).detach().numpy()[0]
            labels = ['non-umami', 'umami']
            string = labels[rst.argmax()]
            ls = []
            ls.append(string)
            for value in rst:
                ls.append('{:.4f}'.format(value))
            total.append(ls)
        except:
            total.append(['error sequence', 'nan', 'nan'])
    return total



model = MLP(n_feats = 2048, n_hiddens = 256, n_tasks = 2)
path = "/pretrained/all_mlp.pth"
total = mlp_bi(model, path, sequences)
print(total)
df = pd.DataFrame(total)
df.columns = ['Odor', 'non-umami', 'umami']
df.insert(0,'Sequence',sequences)
df.to_csv(args.output,index=False,header=True,sep='\t')
