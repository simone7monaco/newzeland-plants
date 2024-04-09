import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer as _KNNImputer
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

import sys
sys.path.append('/home/smonaco/newzeland-plants/GRAPE/')
from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.utils import build_optimizer, get_known_mask, mask_edge
from uci.uci_data import get_data



fern_dataset = pd.read_excel('Simonny dataset final.xlsx')
fern_dataset.index = fern_dataset.Species.str.replace(' ', '_')
fern_dataset = fern_dataset.drop(columns='Species')

# knn imputer which takes both num and cat values but ignores the categorical ones
class KNNImputer(_KNNImputer):
    def fit(self, X, y=None):
        self.num_features = X.select_dtypes(include=[np.number]).columns
        self.cat_features = X.select_dtypes(exclude=[np.number]).columns
        self.num_imputer = _KNNImputer(n_neighbors=self.n_neighbors).fit(X[self.num_features])
        return self
    
    def transform(self, X):
        X_ = X.copy()
        X_[self.num_features] = self.num_imputer.transform(X_[self.num_features])
        return X_


class SimpleImputer(_SimpleImputer):
    def fit(self, X, y=None):
        self.num_features = X.select_dtypes(include=[np.number]).columns
        self.cat_features = X.select_dtypes(exclude=[np.number]).columns
        self.num_imputer = _SimpleImputer(strategy=self.strategy).fit(X[self.num_features])
        return self
    
    def transform(self, X):
        X_ = X.copy()
        X_[self.num_features] = self.num_imputer.transform(X_[self.num_features])
        return X_


class GRAPEImputer:
    def __init__(self, args, device='cuda'):
        self.model = None
        self.args = args
        self.device = device
        impute_hiddens = list(map(int,args.impute_hiddens.split('_')))
        input_dim = args.node_dim * 2
        output_dim = 1

        self.impute_model = MLPNet(input_dim, output_dim,
                                hidden_layer_sizes=impute_hiddens,
                                hidden_activation=args.impute_activation,
                                dropout=args.dropout).to(device)

    def load_data(self, Xy):
        df_X = Xy.drop(columns='Family')
        df_y = Xy.Family
        df_y = pd.Series(pd.Categorical(df_y).codes)
        data = get_data(df_X, df_y, self.args.node_mode, self.args.train_edge, 
                        self.args.split_sample, self.args.split_by, self.args.train_y, self.args.seed)
        return data

    def fit(self, Xy):
        # build data on top of X
        data = self.load_data(Xy)
        self.model = get_gnn(data, self.args)
        self.model.to(self.device)
        trainable_parameters = list(self.model.parameters()) \
                            + list(self.impute_model.parameters())
        print("total trainable_parameters: ",len(trainable_parameters))
        # build optimizer
        scheduler, opt = build_optimizer(self.args, trainable_parameters)

        # train
        self.Train_loss = []
        self.Test_rmse = []
        self.Test_l1 = []
        self.Lr = []

        x = data.x.clone().detach().to(self.device)
        
        all_train_edge_index = data.train_edge_index.clone().detach().to(self.device)
        all_train_edge_attr = data.train_edge_attr.clone().detach().to(self.device)
        all_train_labels = data.train_labels.clone().detach().to(self.device)
        test_input_edge_index = all_train_edge_index
        test_input_edge_attr = all_train_edge_attr
        test_edge_index = data.test_edge_index.clone().detach().to(self.device)
        test_edge_attr = data.test_edge_attr.clone().detach().to(self.device)
        test_labels = data.test_labels.clone().detach().to(self.device)
        
        
        train_edge_index, train_edge_attr, train_labels =\
            all_train_edge_index, all_train_edge_attr, all_train_labels
        print("train edge num is {}, test edge num is input {}, output {}"\
                .format(
                train_edge_attr.shape[0],
                test_input_edge_attr.shape[0], test_edge_attr.shape[0]))
        
        for epoch in range(self.args.epochs):
            self.model.train()
            self.impute_model.train()
            known_mask = get_known_mask(self.args.known, int(train_edge_attr.shape[0] / 2)).to(self.device)
            double_known_mask = torch.cat((known_mask, known_mask), dim=0)
            known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

            opt.zero_grad()
            x_embd = self.model(x, known_edge_attr, known_edge_index)
            pred = self.impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
            pred_train = pred[:int(train_edge_attr.shape[0] / 2),0]
            label_train = train_labels

            loss = F.mse_loss(pred_train, label_train)
            loss.backward()
            opt.step()
            train_loss = loss.item()
            if scheduler is not None:
                scheduler.step(epoch)
            for param_group in opt.param_groups:
                self.Lr.append(param_group['lr'])

            self.model.eval()
            self.impute_model.eval()
            with torch.no_grad():
                x_embd = self.model(x, test_input_edge_attr, test_input_edge_index)
                pred = self.impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
                
                pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
                label_test = test_labels
                mse = F.mse_loss(pred_test, label_test)
                test_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_test, label_test)
                test_l1 = l1.item()
                
                self.Train_loss.append(train_loss)
                self.Test_rmse.append(test_rmse)
                self.Test_l1.append(test_l1)
                print('epoch: ', epoch)
                print('loss: ', train_loss)
                print('test rmse: ', test_rmse)
                print('test l1: ', test_l1)

        self.pred_train = pred_train.detach().cpu().numpy()
        self.label_train = label_train.detach().cpu().numpy()
        self.pred_test = pred_test.detach().cpu().numpy()
        self.label_test = label_test.detach().cpu().numpy()
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.pred_test

    def transform(self, X):
        # TODO: fare meglio, ora ignora X e usa il test set
        return self.pred_test

def main0():
    from sklearn.model_selection import train_test_split
    # instead:
    from argparse import Namespace
    args = Namespace(train_edge=0.7, split_sample=0., split_by='y', 
                    train_y=0.7, node_mode=0, model_types='EGSAGE_EGSAGE_EGSAGE', post_hiddens=None, 
                    norm_embs=None, aggr='mean', node_dim=64, edge_dim=64, concat_states=False,
                    edge_mode=1, gnn_activation='relu', impute_hiddens='64', impute_activation='relu', 
                    epochs=2000, opt='adam', opt_scheduler='none', opt_restart=0, opt_decay_step=100, 
                    opt_decay_rate=0.9, dropout=0., weight_decay=0., lr=0.001, known=0.7, auto_known=False,
                    loss_mode=0, valid=0., seed=42, mode='train')

    # TODO: al momento GRAPE vuole anche la famiglia, gli altri no!
    train, test = train_test_split(fern_dataset, test_size=.3, shuffle=True, random_state=42)

    # default estimators are lgbm classifier and regressor
    grape = GRAPEImputer(args)
    # TODO: managing cat features inside
    grape.fit(fern_dataset.drop(columns=['Separation', 'Habitat']))
    test_imputed = grape.transform(test)

def main():
    float_cols = fern_dataset.columns[fern_dataset.dtypes == 'float']
    def mape_with_mask(y_true, y_pred, mask):
        assert y_true.shape == y_pred.shape and y_true.shape == mask.shape, f"{y_true.shape}, {y_pred.shape}, {mask.shape}"
        results = {c: ((y_true[c][mask[c]] - y_pred[c][mask[c]]).abs() / y_true[c][mask[c]]).mean() for c in mask.columns}
        return pd.Series(results)

    kf = GroupKFold(n_splits=5)

    models = [SimpleImputer(strategy='mean'), KNNImputer()]
    results = pd.DataFrame(np.hstack([
        fern_dataset[float_cols].isna().mean().values.reshape(-1, 1), np.zeros([len(float_cols), len(models)])
        ]), columns=['Missing ratio'] + [m.__class__.__name__ for m in models], index=float_cols).sort_values('Missing ratio')

    for i, (train_index, test_index) in tqdm(enumerate(kf.split(fern_dataset, groups=fern_dataset.Family)), total=kf.get_n_splits()):
        filt_ds = fern_dataset.drop(columns=['Family']).copy()
        # for each column with numerical values, add an extra 15% of missing values

        for col in float_cols:
            filt_ds.loc[filt_ds[filt_ds[col].notna()].sample(n=len(filt_ds)//100*15).index, col] = np.nan

        train, test = filt_ds.iloc[train_index].copy(), filt_ds.iloc[test_index].copy()
        hiddenvals_mask = (filt_ds[float_cols].isna() & fern_dataset[float_cols].notna()).fillna(False).iloc[test_index]
        for model in models:
            if model.__class__.__name__ == 'MissForest':
                model.fit(train.copy(), categorical=filt_ds.columns[filt_ds.dtypes == 'object'])
            else:
                model.fit(train)
            imputed = model.transform(test)
            results.loc[:, model.__class__.__name__] += mape_with_mask(fern_dataset[float_cols].iloc[test_index], imputed[float_cols], hiddenvals_mask)

    results = results.div(kf.get_n_splits())
    results.loc[float_cols].sort_values('Missing ratio')


if __name__ == '__main__':
    main0()