import pandas as pd
import numpy as np
import os
import random
from feature import Graph_smiles
parent_dir = os.path.abspath(os.path.dirname(__file__))
print(parent_dir)

def statistics():
    df = pd.read_csv(parent_dir+"/dataset/train/data.csv", header=0, sep="\t")
    dic = df['Taste'].value_counts().to_dict()
    print(dic)
    
    ls = []
    for index,row in df.iterrows():
        if row['Taste'] == 'umami':
            ls.append(1)
        else:
            ls.append(0)
    print(len(ls))

    df.insert(4,'Label',ls)
    df.to_csv(parent_dir+"/dataset/train/data_labeled.csv",header=True, sep="\t", index=False)


def data_preprocess():
    import rdkit
    import rdkit.Chem as Chem

    df = pd.read_csv(parent_dir+"/dataset/train/data_labeled.csv", header=0, sep="\t")
    print(df['Label'].value_counts())

    # delete can't recognize by rdkit
    none_list=[]
    smileses = []
    for index, row in df.iterrows():
        if Chem.MolFromFASTA(row['Sequence']) is None:
            
            none_list.append(index)
        else:
            smileses.append(Chem.MolToSmiles(Chem.MolFromFASTA(row['Sequence'])))
    df=df.drop(none_list)
    print("rdkit dropping {} unidentified molecules, remain {} molecules".format(len(none_list), len(df)))
    df.insert(5,'Smiles',smileses)

    # delete can't recognize by rdkit
    none_list=[]
    for index, row in df.iterrows():
        try:
            Graph_smiles(Chem.MolToSmiles(Chem.MolFromFASTA(row['Sequence'])))
        except:
            none_list.append(index)
    df=df.drop(none_list)
    print("dlg dropping {} unidentified molecules, remain {} molecules".format(len(none_list), len(df)))
    
    # drop duplicates
    df_new = pd.DataFrame()
    for key in df['Label'].value_counts().to_dict().keys():
        df_ = df.query('Label == @key')
        df_ = df_.drop_duplicates(subset=['Sequence'], keep='first')
        print(key, len(df_))
        df_new = pd.concat([df_new,df_],axis = 0)
    print(len(df_new))

    # no sample
    print(df_new['Label'].value_counts())
    df_new.to_csv(parent_dir+"/dataset/train/cleaned_data.csv", header=True, sep="\t", index=False)
    



if __name__ == "__main__":
    statistics()
    data_preprocess()


