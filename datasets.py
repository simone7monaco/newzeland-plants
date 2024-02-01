import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
re_onehot = lambda value: [10., 0., 1.] if value else [10., 1., 0.] # set each one-hot vector as [10, isFalse, isTrue] (10 is the feature index for categorical features)

from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops

macro_trait_index = {
	0: ['Stature'],
	1: ['PetioleSize'],
	2: ['LeafSize'],
	3: ['InflorescenceSize'],
	4: ['PedicelSize'],
	5: ['FruitSize', 'PappusSize'], # Fruit
	6: ['SeedSize'],
	7: ['FlowerSize', 'CalyxSize', 'SepalSize', 'CorollaSize', 'PetalSize', 'RayFloretsSize', 'DiskFloretSize', 'TubeSize', 'Glumes', 'Lemma', 'LobesSize', 'Lodicules', 'Palea'], # Flower sterile
	8: ['OvarySize', 'StigmaSize', 'StyleSize', 'StamenSize', 'AntherSize', 'UtricleSize'] # Flower fertile
}

class SpecieDataset(InMemoryDataset):
    def __init__(self, root, dataset, genus, edges_gen, indices=None, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        root_parent = Path('graph_dataset')
        root_parent.mkdir(exist_ok=True)
        self.split = root
        self.dataset = dataset
        self.edges_gen = edges_gen
        self.genus = genus
        self.split_indices = dataset.index.isin(dataset.index[indices]) if indices is not None else dataset.index.isin(dataset.index)
        self._add_self_loops = kwargs.get('add_self_loops', False)
        
        super().__init__(root_parent/root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        ...

    def process(self):
        data_raw = self.dataset[self.split_indices].values
        genus_raw = self.genus[self.dataset.index][self.split_indices].values
        data_list = []
        for i, feat in tqdm(enumerate(data_raw), total=len(data_raw)):
            feat = np.stack(feat)
            edges = torch.tensor(self.edges_gen[genus_raw[i]][['level_0', 'level_1']].values).T.long()
            if self._add_self_loops:
                edges = add_self_loops(edges, num_nodes=feat.shape[0])[0]
            # edge_attr = torch.tensor(edges_gen[genus_raw[i]][0].values).float() # TODO: still to add
            data_list.append(Data(x=torch.tensor(feat).float(),
						  edge_index=edges,
                        #   edge_attr=edge_attr,
						  y=torch.tensor(feat[:, 2:3]).float()))
            # TODO: add the edge features
		
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        

def build_dataset(na_threshold=0.7, intermediate_steps=False):
	num_features = pd.read_csv('processed_features_num.csv', index_col=0)
	cat_features = pd.read_excel('Words before and after traits_v2.xlsx', sheet_name='Categorical traits').drop(columns=['FlowerSize'])
	cat_features = cat_features.set_index(cat_features.SpeciesName.str.lower().str.replace(' ', '-')).drop(columns=['SpeciesName'])

	genus = cat_features.Genus
	cat_features = cat_features[cat_features.index.isin(num_features.index)].drop(columns=['Genus', 'Sex', 'Height.m']) # TODO: What is H.m?
	num_features = num_features[num_features.index.isin(cat_features.index)]

	cols_to_drop = num_features.isna().sum()[num_features.isna().sum() > na_threshold*num_features.shape[0]].index
	num_features = num_features.drop(columns=cols_to_drop)


	### Categorical features
	cat_features =cat_features.loc[num_features.index, :]
	ohe = OneHotEncoder(sparse_output=False, drop='if_binary', min_frequency=.02)
	ohe.fit(cat_features)
	cat_features_ohe = pd.DataFrame(ohe.transform(cat_features), columns=ohe.get_feature_names_out(), index=cat_features.index)
	cat_features_ohe = cat_features_ohe.loc[:, cat_features_ohe.nunique() > 1]
	# All categorical features are processed, TODO: consider differently the Family, as Genus?


	### Numerical features
	pure_numerical_features = num_features[[c for c in num_features.columns if not c.endswith('_isabsent')]].drop('Family', axis=1)
	scaler = RobustScaler()
	scaler.fit(pure_numerical_features)

	numerical_features_scaled = pd.DataFrame(scaler.transform(pure_numerical_features), columns=pure_numerical_features.columns, index=pure_numerical_features.index)
	numerical_features_scaled = numerical_features_scaled.fillna(-1)

	trait_index = {trait: k for k, v in macro_trait_index.items() for trait in v}
	def trait_to_feature(traits:pd.Series):
		"""
		traits is a series of a trait for all the species e.g. "AntherSize_l", takes from dataset[trait+'_isabsent'] the corresponding isabsent column
		and returns an array (n_species, 2) with isabsent and the trait
		"""
		isabsent_col = traits.name.split('_')[0] + '_isabsent'
		isabsent = num_features[isabsent_col] if isabsent_col in num_features.columns else [0]*len(num_features)
		
		trait_id = [trait_index[traits.name.split('_')[0]]]*len(num_features)
		return pd.Series(list(np.array([trait_id, isabsent, traits]).T), index=traits.index)

	numerical_features_scaled = numerical_features_scaled.apply(trait_to_feature, axis=0)


	### Edge construction
	genus = genus[pure_numerical_features.index]
	tmp = genus.value_counts()
	mixed_indices = tmp[tmp<10].index
	genus = genus.replace(mixed_indices, 'mixed')

	## pearson correlation between numerical features [BEFORE scaling]
	all_features = pure_numerical_features.join(cat_features_ohe).join(genus)
	edge_threshold = 0.5
	first_cat_node = len(pure_numerical_features.columns)

	corr_matrices = {}
	edges_gen = {}
	for gen, gen_allfeatures in all_features.groupby('Genus'):
		pears_corr = gen_allfeatures.drop(columns=['Genus']).corr(method='pearson')
		nodes_names = pears_corr.columns

		edges = pears_corr.abs().unstack().sort_values(ascending=False)
		edges = edges[edges.gt(edge_threshold) & edges.lt(1)].reset_index()

		edges[['level_0', 'level_1']] = edges[['level_0', 'level_1']].map(lambda name: nodes_names.get_loc(name))
		edges_gen[gen] = edges
		corr_matrices[gen] = pears_corr


	skf = StratifiedKFold(n_splits=5, shuffle=True)
	train_idx, val_idx = next(skf.split(num_features,
										cat_features_ohe[[c for c in cat_features_ohe.columns if c.startswith('Family')]].apply(np.argmax, axis=1)
										))
     
	full_dataset = numerical_features_scaled.join(cat_features_ohe.map(re_onehot))
	if intermediate_steps:
		return num_features, numerical_features_scaled, cat_features_ohe, edges_gen, genus, train_idx, val_idx, corr_matrices
	return full_dataset, edges_gen, genus, train_idx, val_idx
