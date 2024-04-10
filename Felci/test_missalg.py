import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer as _KNNImputer
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from tqdm.auto import tqdm, trange

import torch
import torch.nn.functional as F

import sys
sys.path.append('/home/smonaco/newzeland-plants/GRAPE/')
from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.utils import build_optimizer, get_known_mask, mask_edge
from uci.uci_data import get_data


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

    def load_data(self, Xy, known_mask=None):
        df_X = Xy.drop(columns='Family')
        df_y = Xy.Family
        df_y = pd.Series(pd.Categorical(df_y).codes)
        data, self.Xscaler = get_data(df_X, df_y, self.args.node_mode, self.args.train_edge, 
                                      self.args.split_sample, self.args.split_by, self.args.train_y, 
                                      self.args.seed, return_scaler=True, train_edge_mask=known_mask)
        return data

    def fit_transform(self, Xy, known_mask=None, verbose=False):
        """
        Xy: pandas dataframe with the last column being the target ('Family' in this case)
        split_indices: list of train and test indices of the initial tabular dataset
        """
        data = self.load_data(Xy, known_mask=torch.tensor(known_mask).bool())

        self.model = get_gnn(data, self.args)
        self.model.to(self.device)
        trainable_parameters = list(self.model.parameters()) \
                            + list(self.impute_model.parameters())
        if verbose:
            print("total trainable_parameters: ", len(trainable_parameters))
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
        if verbose:
            print("train edge num is {}, test edge num is input {}, output {}"\
                    .format(
                    train_edge_attr.shape[0],
                    test_input_edge_attr.shape[0], test_edge_attr.shape[0])) # TODO: why? train edge num is 8746, test edge num is input 8746, output 3674
        bar = trange(self.args.epochs)
        for epoch in bar:
            self.model.train()
            self.impute_model.train()
            known_mask_ = get_known_mask(self.args.known, int(train_edge_attr.shape[0] / 2)).to(self.device)
            double_known_mask = torch.cat((known_mask_, known_mask_), dim=0)
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
                mse = F.mse_loss(pred_test[~label_test.isnan()], label_test[~label_test.isnan()])
                test_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_test, label_test)
                test_l1 = l1.item()
                
                self.Train_loss.append(train_loss)
                self.Test_rmse.append(test_rmse)
                self.Test_l1.append(test_l1)
                bar.set_postfix({'loss': train_loss, 'test rmse': test_rmse, 'test l1': test_l1})

        # plot losses in grape_losses.png
        fig = plt.figure()
        plt.plot(np.array(self.Train_loss)**.5, label='train rmse')
        plt.plot(self.Test_rmse, label='test rmse')
        plt.legend()
        fig.savefig('grape_losses.png')
        
        self.pred_train = pred_train.detach().cpu().numpy()
        self.label_train = label_train.detach().cpu().numpy()
        self.pred_test = pred_test.detach().cpu().numpy()
        self.label_test = label_test.detach().cpu().numpy()

        pred_table = Xy.drop(columns='Family').values.flatten()
        pred_table[~known_mask] = self.pred_test
        pred = self.Xscaler.inverse_transform(pred_table.reshape(-1, len(Xy.columns) - 1))
        return pd.DataFrame(pred, columns=Xy.columns.drop('Family'), index=Xy.index)

from argparse import Namespace
args = Namespace(train_edge=0.7, split_sample=0., split_by='y', 
                 train_y=0.7, node_mode=0, model_types='EGSAGE_EGSAGE_EGSAGE', post_hiddens=None, 
                 norm_embs=None, aggr='mean', node_dim=64, edge_dim=64, concat_states=False,
                 edge_mode=1, gnn_activation='relu', impute_hiddens='64', impute_activation='relu', 
                 epochs=200, opt='adam', opt_scheduler='none', opt_restart=0, opt_decay_step=100, 
                 opt_decay_rate=0.9, dropout=0., weight_decay=0., lr=0.001, known=0.7, auto_known=False,
                 loss_mode=0, valid=0., seed=42, mode='train')


# TODO: better node_mode=1 (sample is one of the 'hots')
def main():
    fern_dataset = pd.read_excel('Simonny dataset final.xlsx')
    fern_dataset.index = fern_dataset.Species.str.replace(' ', '_')
    fern_dataset = fern_dataset.drop(columns='Species')
    
    num_cols = fern_dataset.select_dtypes(exclude='object').columns
    cat_cols = fern_dataset.select_dtypes(include='object').columns.drop('Family') # Family is the target / it is ignored
    fern_dataset = pd.get_dummies(fern_dataset, columns=cat_cols, drop_first=True)
    fern_dataset = fern_dataset.astype({c: 'float' for c in fern_dataset.columns if c != 'Family'})
    
    def mape_with_mask(y_true, y_pred, mask):
        assert y_true.shape == y_pred.shape and y_true.shape == mask.shape, f"{y_true.shape}, {y_pred.shape}, {mask.shape}"
        results = {c: ((y_true[c][mask[c]] - y_pred[c][mask[c]]).abs() / y_true[c][mask[c]]).mean() for c in mask.columns}
        return pd.Series(results)

    cv = KFold(n_splits=5) # intended to be applied on the whole dataset instances, if they are not NaN

    models = [GRAPEImputer(args), SimpleImputer(strategy='mean'), KNNImputer()]
    results = pd.DataFrame(np.hstack([
        fern_dataset[num_cols].isna().mean().values.reshape(-1, 1), np.zeros([len(num_cols), len(models)])
        ]), columns=['Missing ratio'] + [m.__class__.__name__ for m in models], index=num_cols).sort_values('Missing ratio')

    known_values_idx = np.arange(fern_dataset.shape[0] * (fern_dataset.shape[1] - 1))
    target_values_mask = fern_dataset.drop(columns='Family').notna().copy()
    target_values_mask[[c for c in target_values_mask.columns if c.split('_')[0] in cat_cols]] = False
    known_values_idx = known_values_idx[target_values_mask.values.flatten()]
    for i, (train_index, test_index) in tqdm(enumerate(cv.split(known_values_idx)), total=cv.get_n_splits()):
        values = fern_dataset.drop(columns=['Family']).values.flatten()
        values[test_index] = np.nan
        filt_ds = fern_dataset.copy()
        filt_ds.loc[:, filt_ds.columns != 'Family'] = values.reshape(-1, len(filt_ds.columns) - 1)

        # known_mask: boolean mask of the test indices
        known_mask = ~np.isnan(values)

        hiddenvals_mask = (filt_ds[num_cols].isna() & fern_dataset[num_cols].notna()).fillna(False)  # remove already nan vals
        # TODO: ottenere lo stesso da una maschera completa dei nan e nanmean, etc.

        for model in models:
            if model.__class__.__name__ == 'GRAPEImputer':
                # TODO: gestire internamente cat features (senza cancellarle)
                imputed = model.fit_transform(fern_dataset, known_mask=known_mask)
            else:
                imputed = model.fit_transform(filt_ds)
            results.loc[:, model.__class__.__name__] += mape_with_mask(fern_dataset[num_cols], imputed[num_cols], hiddenvals_mask)

    results = results.div(cv.get_n_splits())
    print(results.loc[num_cols].sort_values('Missing ratio'))


if __name__ == '__main__':
    main()