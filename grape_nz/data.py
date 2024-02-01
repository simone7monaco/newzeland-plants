import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing

import torch
import random
import numpy as np
import pdb

from utils import get_known_mask, get_fixed_known_mask, mask_edge

def create_node(df, mode, family=None):
    if mode == 0: # onehot feature node, all 1 sample node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol))
        feature_node[np.arange(ncol), feature_ind] = 1
        sample_node = [[1]*ncol for i in range(nrow)]
        node = sample_node + feature_node.tolist()
    elif mode == 1: # onehot sample and feature node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol+1))
        feature_node[np.arange(ncol), feature_ind+1] = 1
        sample_node = np.zeros((nrow,ncol+1))
        sample_node[:,0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    elif mode == 2: # onehot sample and feature node
        raise NotImplementedError
        # family is a pd.Series with the family name of each sample, should be converted
        # to a one-hot encoding one column per family
        # TODO: place under-represented families in a macro-category "other"
        family_onehot = pd.get_dummies(family)
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol+families_onehot.shape[1]))
        feature_node[np.arange(ncol), feature_ind] = 1
        sample_node = np.zeros((nrow,ncol+families_onehot.shape[1]))
        sample_node[:,:ncol] = 1
        sample_node[:,ncol:] = families_onehot.values
        node = sample_node.tolist() + feature_node.tolist()
    return node

def create_edge(df):
    n_row, n_col = df.shape
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col # obj
        edge_end = edge_end + list(n_row+np.arange(n_col)) # att    
    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start
    return (edge_start_new, edge_end_new)

def create_edge_attr(df):
    nrow, ncol = df.shape
    edge_attr = []
    for i in range(nrow):
        for j in range(ncol):
            edge_attr.append([float(df.iloc[i,j])])
    edge_attr = edge_attr + edge_attr
    return edge_attr

def get_data(df_X, df_X_cat, args, normalize=True):

    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)

    #remove edges if the edge_attr is nan --> they are not present in the dataset
    edge_index = edge_index[:, ~torch.isnan(edge_attr).squeeze()]
    edge_attr = edge_attr[~torch.isnan(edge_attr).squeeze()]

    node_init = create_node(df_X, args.node_mode, df_X_cat) 
    x = torch.tensor(node_init, dtype=torch.float)
    
    #set seed to fix known/unknwon edges
    torch.manual_seed(args.seed)
    #keep train_edge_prob of all edges
    if args.cross_validation:
        # 5 splits is equivalent to a train_edge probabiliy of 0.8
        train_edge_mask = get_fixed_known_mask(int(edge_attr.shape[0]/2), n_splits=5, split=args.split, seed=args.seed)
    else:
        train_edge_mask = get_known_mask(args.train_edge, int(edge_attr.shape[0]/2))
    # store the list of known edges
    with open(osp.join(args.log_path, 'unknown_edges.txt'), 'w') as f:
        f.write(str(np.arange(train_edge_mask.shape[0])[~train_edge_mask.numpy()].tolist()))
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)

    #mask edges based on the generated train_edge_mask
    #train_edge_index is known, test_edge_index in unknwon, i.e. missing

    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                double_train_edge_mask, True)
    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2),0]
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~double_train_edge_mask, True)
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2),0]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,
            train_edge_mask=train_edge_mask,train_labels=train_labels,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,
            test_edge_mask=~train_edge_mask,test_labels=test_labels, 
            df_X=df_X,
            edge_attr_dim=train_edge_attr.shape[-1],
            user_num=df_X.shape[0]
            )
    return data

def load_data(args):
    # TODO: aggiungere feat. categoriche.Come vengono gestite nell'imputing?
    df_X = pd.read_csv('../processed_features_num.csv', index_col=0)
    df_X = df_X.drop(columns=[c for c in df_X.columns if c.endswith('_isabsent') or c.endswith('_w')])
    cols_to_drop = df_X.isna().sum()[df_X.isna().sum() > args.feat_nathr*df_X.shape[0]].index
    print(f'Columns to drop (thr: {args.feat_nathr}): {cols_to_drop}')
    df_X = df_X.drop(columns=cols_to_drop)

    df_X = df_X.reset_index(drop=True)
    df_X_cat = df_X.Family
    df_X = df_X.drop(columns=['Family']).T.reset_index(drop=True).T

    data = get_data(df_X, df_X_cat, args=args)
    # args.split_sample := 0
    return data


