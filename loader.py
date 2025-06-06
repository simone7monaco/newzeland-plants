import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import itertools
import pandas as pd
from Bio import Phylo
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import GroupKFold, KFold

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import KNNGraph, BaseTransform
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import subgraph, bipartite_subgraph


class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes using min-max scaling."""
    def __init__(self, normalizations=None):
        # self.normalizations = normalizations or {
        #     'spatial/x': 'sincos', # They are latitude and longitude
        #     'spatial/global_data': 'z',
        #     'species/x': 'logz', # always positive and sometimes skewed
        #     'species/y': 'logz', 
        #     'species/x_phylo': None, # x_phylo is a vector of embeddings, so no normalization
        #     'spatial-spatial/edge_attr': 'logz',
        #     'species-species/edge_attr': 'z',
        #     'spatial-species/edge_attr': 'z',
        # }
        self.normalizations = normalizations or {
            'spatial_x': 'sincos', # They are latitude and longitude
            'spatial_global_data': 'z',
            'species_x': 'logz', # always positive and sometimes skewed
            'species_y': 'logz', 
            'species_x_phylo': None, # x_phylo is a vector of embeddings, so no normalization
            'spatial_spatial_edge_attr': 'logz',
            'species_species_edge_attr': 'z',
            'spatial_species_edge_attr': 'z',
        }
        self.props = {}

    def forward(self, data: Data):
        for k_norm in self.normalizations:
            if self.normalizations[k_norm] == 'sincos':
                data[k_norm] = data[k_norm] * np.pi / 180
                data[k_norm] = torch.stack([torch.sin(data[k_norm][:, 0]), torch.cos(data[k_norm][:, 0]),
                                          torch.sin(data[k_norm][:, 1]), torch.cos(data[k_norm][:, 1])], dim=1)
            elif self.normalizations[k_norm] in ['logz', 'z']:
                # log normalization and/or z normalization
                if self.normalizations[k_norm] == 'logz':
                    data[k_norm] = torch.log(data[k_norm] + 1e-6)
                mean = data[k_norm].mean(dim=0)
                std = data[k_norm].std(dim=0)
                data[k_norm] = (data[k_norm] - mean) / std
                self.props[k_norm] = {'mean': mean, 'std': std}
            elif self.normalizations[k_norm] is None:
                pass
            else:
                raise ValueError(f"Unknown normalization: {self.normalizations[k_norm]}")
        return data
    
    def inverse(self, data):
        data = data.clone()
        for k_norm in self.normalizations:
            if self.normalizations[k_norm] == 'sincos':
                lat = torch.atan2(data[k_norm][:, 0], data[k_norm][:, 1]) * 180 / np.pi
                lon = torch.atan2(data[k_norm][:, 2], data[k_norm][:, 3]) * 180 / np.pi
                data[k_norm] = torch.stack([lat, lon], dim=1)
            elif self.normalizations[k_norm] in ['logz', 'z']:
                # log normalization and/or z normalization
                data[k_norm] = data[k_norm] * self.props[k_norm]['std'] + self.props[k_norm]['mean']
                if self.normalizations[k_norm] == 'logz':
                    data[k_norm] = torch.exp(data[k_norm]) - 1e-6
        return data
    
    # def forward(self, data: HeteroData):
    #     for k_norm in self.normalizations:
    #         n, k = k_norm.split('/')
    #         if len(n.split('-')) > 1:
    #             n = tuple(n.split('-'))
            
    #         if self.normalizations[k_norm] == 'sincos':
    #             data[n][k] = data[n][k] * np.pi / 180
    #             data[n][k] = torch.stack([torch.sin(data[n][k][:, 0]), torch.cos(data[n][k][:, 0]),
    #                                       torch.sin(data[n][k][:, 1]), torch.cos(data[n][k][:, 1])], dim=1)
    #         elif self.normalizations[k_norm] in ['logz', 'z']:
    #             # log normalization and/or z normalization
    #             if self.normalizations[k_norm] == 'logz':
    #                 data[n][k] = torch.log(data[n][k] + 1e-6)
    #             mean = data[n][k].mean(dim=0)
    #             std = data[n][k].std(dim=0)
    #             data[n][k] = (data[n][k] - mean) / std
    #             self.props[k_norm] = {'mean': mean, 'std': std}
    #         elif self.normalizations[k_norm] is None:
    #             pass
    #         else:
    #             raise ValueError(f"Unknown normalization: {self.normalizations[k_norm]}")
    #     return data
    
    # def inverse(self, data, warn=True):
    #     data = data.clone()
    #     for k_norm in self.normalizations:
    #         n, k = k_norm.split('/')
    #         if len(n.split('-')) > 1:
    #             n = tuple(n.split('-'))
            
    #         if n not in (data.node_types + data.edge_types) or k not in data[n].keys():
    #             if warn:
    #                 print(f"Warning: {n} or {n}/{k} not in data")
    #             continue
    #         if self.normalizations[k_norm] == 'sincos':
    #             lat = torch.atan2(data[n][k][:, 0], data[n][k][:, 1]) * 180 / np.pi
    #             lon = torch.atan2(data[n][k][:, 2], data[n][k][:, 3]) * 180 / np.pi
    #             data[n][k] = torch.stack([lat, lon], dim=1)
    #         elif self.normalizations[k_norm] in ['logz', 'z']:
    #             # log normalization and/or z normalization
    #             data[n][k] = data[n][k] * self.props[k_norm]['std'] + self.props[k_norm]['mean']
    #             if self.normalizations[k_norm] == 'logz':
    #                 data[n][k] = torch.exp(data[n][k]) - 1e-6
    #     return data
    

class NZData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'species_species_edge_index':
            return self.species_num_nodes
        elif key == 'spatial_spatial_edge_index':
            return self.spatial_num_nodes
        elif key == 'spatial_species_edge_index':
            return torch.tensor([[self.spatial_num_nodes], [self.species_num_nodes]])
        return super().__inc__(key, value, *args, **kwargs)


class FernDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.traits_all = self.traits_df(root)
        self.y_index = np.arange(len(self.traits_all.columns))[[not (c.startswith('Habitat') or c.startswith('Family') or c.startswith('Hybridisation')) for c in self.traits_all.columns]]

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
    
    @staticmethod
    def load_raster(f, grid_step=.1):
        raster = xr.open_dataarray(f)
        current_step = (raster.x[1] - raster.x[0]).values # degrees
        downsample = round(grid_step / current_step)
        raster = raster.coarsen(x=downsample, y=downsample, boundary="trim").sum().fillna(0)
        return raster
    
    @staticmethod
    def traits_df(root, dummy_threshold=0.05):
        traits_mean = pd.read_excel(root / 'Traits.xlsx', sheet_name='Mean').set_index('Species')
        traits_var = pd.read_excel(root / 'Traits.xlsx', sheet_name='Var').drop(columns=['Family', 'Habitat', 'Hybridisation']).set_index('Species')
        traits_std = traits_var.apply(np.sqrt).rename(columns=lambda x: x.replace('Var', 'Std'))

        traits_all = traits_mean.join(traits_std)
        traits_all = traits_all.drop(index='Hymenophyllum_falklandicum') # missing information
        for cl_feat in ['Habitat', 'Family']:
            for cl in traits_all[cl_feat].unique():
                if traits_all[cl_feat].eq(cl).sum() < len(traits_all) * dummy_threshold:
                    traits_all[cl_feat] = traits_all[cl_feat].replace({cl: 'Other'})
        traits_all = pd.get_dummies(traits_all, drop_first=True)
        return traits_all

    @property
    def raw_dir(self):
        return Path(self.root) / 'Distribution layers'
    
    @property
    def raw_file_names(self):
        dist_files = [f'{f}_distribution.tif' for f in self.traits_all.index]
        return dist_files

    @property
    def processed_file_names(self):
        return ['heterodata.pt']

    def download(self):
        raise RuntimeError('Dataset not found.')

    def load_complete(self, data_path):
        comp_rasters = []
        df_path = (Path(self.root) / "complete layers"/"{data_path}_space_df.csv")
        if df_path.exists():
            return pd.read_csv(df_path, index_col=0)
        
        for f in tqdm(list((Path(self.root) / "complete layers"/data_path).rglob('*.tif')), desc=f'Loading {data_path}'):
            raster = self.load_raster(f)
            filt_raster = raster.interp(x=self.space_df.x.values, y=self.space_df.y.values, method='nearest')
            for band in filt_raster.band.values:
                filt_raster = filt_raster.sel(band=band).to_pandas().reset_index().melt(id_vars=['y'])
                feat_name = f.stem if band == 0 else f"{f.stem}_{band}"
                filt_raster = filt_raster.rename(columns={'value': feat_name})
                # TODO: va bene la media per tutte le variabili?
                comp_rasters.append(self.space_df.set_index(['x', 'y']).join(filt_raster.set_index(['x', 'y'])).groupby(['cluster']).mean()[feat_name])
                if feat_name == 'NZ population density layer':
                    comp_rasters[-1] = np.log1p(comp_rasters[-1])
        df = pd.concat(comp_rasters, axis=1)
        df.to_csv(df_path)
        return df

    def get_species_graph(self):
        tree_file = self.root/"grafted_tree.nwk"
        tree = Phylo.read(tree_file, "newick")

        G = nx.Graph()
        for clade in tree.get_nonterminals():  # Internal nodes (non-leaf)
            if clade.name is None:
                clade.name = f"temp_{id(clade)}"
            for child in clade.clades:
                if child.name is None:
                    child.name = f"temp_{id(child)}"
                G.add_edge(clade.name, child.name, weight=child.branch_length)
        
        from node2vec import Node2Vec
        node2vec = Node2Vec(G, dimensions=16, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Create node embeddings
        node_embeddings = {node: model.wv[node] for node in G.nodes()}

        leaf_nodes = [leaf.name for leaf in tree.get_terminals()]

        G_species = nx.Graph()
        for species1, species2 in itertools.combinations(leaf_nodes, 2):
            dist = tree.distance(species1, species2)  # Compute phylogenetic distance
            G_species.add_edge(species1, species2, weight=dist)

        median = np.median([d["weight"] for u, v, d in G_species.edges(data=True)])
        print(f"Median distance: {median}\n")
        sym_threshold = median

        # Prune the graph by removing edges with weight > sym_threshold
        edges_to_remove = [(u, v) for u, v, d in G_species.edges(data=True) if d["weight"] > sym_threshold]
        G_species.remove_edges_from(edges_to_remove)

        G_species.remove_nodes_from(list(set(G_species.nodes).difference(self.traits_all.index)))

        for node in G_species.nodes():
            G_species.nodes[node]["x"] = node_embeddings[node]
        self.species_graph = from_networkx(G_species)  # Include node attributes
        self.species_graph.node_names = [n for n in G_species.nodes()]  # Map node names to indices

        # Add nodes not present in the ph_tree
        for node in self.traits_all.index.difference(self.species_graph.node_names):
            self.species_graph.node_names.append(node)
            self.species_graph.x = torch.cat([self.species_graph.x, torch.zeros(1, self.species_graph.num_features)], dim=0)

        self.species_graph.traits_nanmask = torch.tensor(self.traits_all.loc[self.species_graph.node_names].isna().values, dtype=torch.bool)
        self.species_graph.traits = torch.tensor(self.traits_all.loc[self.species_graph.node_names].fillna(0).astype(np.float32).values, dtype=torch.float32)
        return self.species_graph

    def get_spatial_graph(self, n_clusters=50, k=6):
        all_occurrences = self.load_raster(Path(self.root)/"Distribution layers/_all_species_distributions_with species info.tif")
        self.space_df = all_occurrences.isel(band=0).to_pandas().reset_index().melt(id_vars=['y']).rename(columns={'value': 'occurrence'})
        self.space_df.x = self.space_df.x.astype(float)
        self.space_df.y = self.space_df.y.astype(float)
        self.space_df = self.space_df[self.space_df.occurrence > 0].drop(columns='occurrence').reset_index(drop=True)
        
        self.space_clustering = SpectralClustering(n_clusters=n_clusters, random_state=0).fit(self.space_df[['x', 'y']])
        self.space_df['cluster'] = self.space_clustering.labels_
        self.spatial_graph = Data(pos=torch.tensor(self.space_df.groupby('cluster').agg({'x': 'mean', 'y': 'mean'}).values).float())
        
        self.spatial_graph = KNNGraph(k=k)(self.spatial_graph)
        self.spatial_graph.edge_attr = torch.norm(self.spatial_graph.pos[self.spatial_graph.edge_index[0]] - self.spatial_graph.pos[self.spatial_graph.edge_index[1]], dim=1).unsqueeze(1)
        return self.spatial_graph

    def get_distribution(self, dist_path):
        ar = self.load_raster(dist_path)

        # give a label to any non zero and non nan value and then aggregate on this dimension
        ar = ar.where(ar > 0, 0)
        ar = ar.where(ar.isnull(), 1)

    def process(self):
        species_graph = self.get_species_graph()
        spatial_graph = self.get_spatial_graph()
        clim_rasters = self.load_complete('climatic layers')
        population_elevation_rasters = self.load_complete('population density and elevation layer')
        soil_rasters = self.load_complete('Soil NZ layers')

        self.global_data = pd.concat([clim_rasters, population_elevation_rasters, soil_rasters], axis=1)
        index_space_specie_path = Path(self.root) / 'index_space_specie.csv'
        if index_space_specie_path.exists():
            index_space_specie = pd.read_csv(index_space_specie_path)
        else:
            index_space_specie = pd.DataFrame()
            for f in tqdm(self.raw_paths, desc='Processing distribution layers'):
                raster = self.load_raster(f)
                species_name = Path(f).stem.rsplit('_', 1)[0]
                # raster = raster.sel(x=space_df.x.values, y=space_df.y.values, method='nearest')
                occurence_df = raster.isel(band=0).to_pandas().reset_index().melt(id_vars=['y']).rename(columns={'value': 'occurrence'}).astype(np.float32)
                occurence_df = occurence_df[occurence_df.occurrence > 0]
                # for each x, y; find the closest point in the space_df
                occurence_df[['x', 'y']] = occurence_df[['x', 'y']].apply(lambda x: self.space_df.iloc[(self.space_df[['x', 'y']] - x).pow(2).sum(1).idxmin()][['x', 'y']], axis=1)
                occurence_df = occurence_df.set_index(['x', 'y']).join(self.space_df.set_index(['x', 'y']))
                index_space_specie = pd.concat([occurence_df.groupby('cluster').sum().reset_index().assign(species=species_name),
                                                index_space_specie,], axis=0, ignore_index=True)
                
            index_space_specie['species_idx'] = index_space_specie.species.apply(lambda x: species_graph.node_names.index(x))
            index_space_specie.to_csv(index_space_specie_path, index=False)
            
        bip_edge_index = torch.tensor(index_space_specie[['cluster', 'species_idx']].values.T, dtype=torch.long)
        bip_edge_attr = torch.tensor(index_space_specie.occurrence.values, dtype=torch.float32).unsqueeze(1)

        data_all = NZData(
        species_x=species_graph.traits,
        species_names=species_graph.node_names,
        species_x_phylo=species_graph.x,
        traits_nanmask=species_graph.traits_nanmask[:, self.y_index],
        spatial_x=spatial_graph.pos,
        spatial_pos=spatial_graph.pos, # repeated to stay un-normalized
        spatial_global_data=torch.tensor(self.global_data.values, dtype=torch.float32),
        species_species_edge_index=species_graph.edge_index,
        species_species_edge_attr=species_graph.weight,
        spatial_spatial_edge_index=spatial_graph.edge_index,
        spatial_spatial_edge_attr=spatial_graph.edge_attr,
        spatial_species_edge_index=bip_edge_index,
        spatial_species_edge_attr=bip_edge_attr,
        species_num_nodes=species_graph.num_nodes,
        spatial_num_nodes=spatial_graph.num_nodes,
        )
        data_all['species_y'] = species_graph.traits[:, self.y_index]*~data_all.traits_nanmask


        # data_all = HeteroData()
        # data_all.species_x = species_graph.traits
        # data_all['species'].names = species_graph.node_names
        # data_all.species_x_phylo = species_graph.x
        # data_all['species'].traits_nanmask = species_graph.traits_nanmask[:, self.y_index]
        # data_all['species'].names = species_graph.node_names
        # data_all['species'].y = species_graph.traits[:, self.y_index]*~data_all['species'].traits_nanmask
        # data_all.spatial_x = spatial_graph.pos
        # data_all.spatial_pos = spatial_graph.pos # repeated to stay un-normalized
        # data_all.spatial_global_data = torch.tensor(self.global_data.values, dtype=torch.float32)
        # data_all['species', 'connects_to', 'species'].edge_index = species_graph.edge_index
        # data_all['species', 'connects_to', 'species'].edge_attr = species_graph.weight
        # data_all['spatial', 'connects_to', 'spatial'].edge_index = spatial_graph.edge_index
        # data_all['spatial', 'connects_to', 'spatial'].edge_attr = spatial_graph.edge_attr
        # data_all['spatial', 'contains', 'species'].edge_index = bip_edge_index
        # data_all['spatial', 'contains', 'species'].edge_attr = bip_edge_attr
        self.save([data_all], self.processed_paths[0])
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # data.y = data.x_species[:, self.y_index]*~data.x_species_traits_nanmask
        return data
        

def data_split(data, test_size=0.3, k=0, seed=42):
    G = nx.Graph()
    G.add_nodes_from(range(data.species_num_nodes))
    # G.add_edges_from(data.edge_index_species.T.numpy())
    G.add_edges_from(data.species_species_edge_index.T.numpy())
    communities = list(nx.algorithms.community.louvain_communities(G))

    cs = []
    for i, community in enumerate(communities):
        cs.extend(((node, i) for node in community))

    cs = pd.DataFrame(cs, columns=['species_idx', 'community'])
    cs.loc[cs.community.groupby(cs.community).transform('size').eq(1), 'community'] = cs.community.max() + 1

    # split the bigger communities into smaller ones (if they are bigger than 1/5 of the dataset, slit by tiles of 1/5 of the community size)
    for i, community in cs.groupby('community'):
        if len(community) > len(cs) / 5:
            cs.loc[community.index, 'community'] = (community.species_idx // (len(community) // 5)).astype(int) + cs.community.max() + 1

    splitter = KFold(n_splits=5, random_state=seed, shuffle=True)
    splits = [s for s in splitter.split(G.nodes())]
    #splits = [s for s in splitter.split(G.nodes(), groups=cs.community)]
    train_nodes, test_nodes = splits[k]

    train_mask = torch.zeros(data.species_num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.species_num_nodes, dtype=torch.bool)

    train_mask[train_nodes] = True
    test_mask[test_nodes] = True

    if train_mask.sum() < test_mask.sum():
        # print(f'Warning: Swapped train and test masks ({train_mask.sum()} < {test_mask.sum()})')
        train_mask, test_mask = test_mask, train_mask
    data.train_mask = train_mask
    data.test_mask = test_mask

    train_data = data.clone()
    test_data = data.clone()
    for attr in ['x', 'y', 'x_phylo',]:
        train_data[f'species_{attr}'] = train_data[f'species_{attr}'][train_mask]
        test_data[f'species_{attr}'] = test_data[f'species_{attr}'][test_mask]
    train_data.traits_nanmask = train_data.traits_nanmask[train_mask]
    test_data.traits_nanmask = test_data.traits_nanmask[test_mask]
    train_data.species_num_nodes = train_mask.sum().item()
    test_data.species_num_nodes = test_mask.sum().item()
    

    for data_split, mask in zip([train_data, test_data], [train_mask, test_mask]):
        data_split.species_species_edge_index, data_split.species_species_edge_attr = subgraph(
            mask, edge_index=data.species_species_edge_index,
            edge_attr=data.species_species_edge_attr,
            relabel_nodes=True
            )
        data_split.spatial_species_edge_index, data_split.spatial_species_edge_attr = bipartite_subgraph(
            (torch.ones(data.spatial_num_nodes, dtype=torch.bool), mask), 
            edge_index=data.spatial_species_edge_index, edge_attr=data.spatial_species_edge_attr, relabel_nodes=True
            )
    return train_data, test_data

if __name__ == '__main__':
    norm_transform = NormalizeFeatures()
    data = FernDataset(Path('data/Ferns'))[0]
    normed_data = norm_transform(data)
    re_unnormed_data = norm_transform.inverse(normed_data)

    for k in norm_transform.normalizations:
        if not torch.allclose(data[k], re_unnormed_data[k]):
            print(f"Normalization failed for {k} (max diff: {torch.max(torch.abs(data[k] - re_unnormed_data[k]))})")
        # n, k = k.split('/')
        # if len(n.split('-')) > 1:
        #     n = n.split('-')
        # if not torch.allclose(data[n][k], re_unnormed_data[n][k]):
        #     print(f"Normalization failed for {n}: {k} (max diff: {torch.max(torch.abs(data[n][k] - re_unnormed_data[n][k]))})")

    