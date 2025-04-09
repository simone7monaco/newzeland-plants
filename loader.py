import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import itertools
import pandas as pd
from Bio import Phylo
import networkx as nx
from node2vec import Node2Vec
from sklearn.cluster import SpectralClustering

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph, BaseTransform
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import subgraph, bipartite_subgraph


class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes using min-max scaling."""
    def __init__(self, attrs=None):
        self.attrs = attrs or ['x_species', 'x_species_phylo', 'x_spatial', 'global_data', 
                               'edge_attr_species', 'edge_attr_spatial', 'bip_edge_attr']
        self.min_max = {}

    def forward(self, data):
        assert all(attr in data for attr in self.attrs), 'Not all attributes are present in the data.'
        for attr in self.attrs:
            if attr not in self.min_max:
                self.min_max[attr] = (data[attr].min(dim=0)[0], data[attr].max(dim=0)[0])
            min_val, max_val = self.min_max[attr]
            data[attr] = (data[attr] - min_val) / (max_val - min_val)
        return data
    
    def inverse(self, data):
        for attr in self.attrs:
            if attr in data:
                min_val, max_val = self.min_max[attr]
                data[attr] = data[attr] * (max_val - min_val) + min_val
        return data


class FernDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.traits_all = self.traits_df(root)
        self.y_index = np.arange(len(self.traits_all.columns))[[not (c.startswith('Habitat') or c.startswith('Family')) for c in self.traits_all.columns]]

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
        return ['data.pt']

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
        # Phylo.draw_ascii(tree)

        G = nx.Graph()
        for clade in tree.get_nonterminals():  # Internal nodes (non-leaf)
            if clade.name is None:
                clade.name = f"temp_{id(clade)}"
            for child in clade.clades:
                if child.name is None:
                    child.name = f"temp_{id(child)}"
                G.add_edge(clade.name, child.name, weight=child.branch_length)
        
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

        data_all = Data(
            x_species=species_graph.traits,
            x_species_phylo=species_graph.x,
            x_species_traits_nanmask=species_graph.traits_nanmask[:, self.y_index],
            x_spatial=spatial_graph.pos,
            species_names=species_graph.node_names,
            edge_index_species=species_graph.edge_index,
            edge_index_spatial=spatial_graph.edge_index,
            edge_attr_species=species_graph.weight,
            edge_attr_spatial=spatial_graph.edge_attr,
            bip_edge_index=bip_edge_index,
            bip_edge_attr=bip_edge_attr,
            global_data=torch.tensor(self.global_data.values, dtype=torch.float32),
            num_nodes=species_graph.num_nodes,
        )
        self.save([data_all], self.processed_paths[0])
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.y = data.x_species[:, self.y_index]*~data.x_species_traits_nanmask
        return data


def data_split(data, test_size=0.3, k=0):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index_species.T.numpy())
    communities = list(nx.algorithms.community.louvain_communities(G))

    # Flatten communities into node indices
    node_community = [np.argmax([node in comm for comm in communities]) for node in G.nodes()]

    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(test_size=test_size, random_state=42, n_splits=3)
    splits = [s for s in splitter.split(G.nodes(), groups=node_community)]
    train_nodes, test_nodes = splits[k]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_nodes] = True
    test_mask[test_nodes] = True

    if train_mask.sum() < test_mask.sum():
        print(f'Warning: Swapped train and test masks ({train_mask.sum()} < {test_mask.sum()})')
        train_mask, test_mask = test_mask, train_mask
    data.train_mask = train_mask
    data.test_mask = test_mask

    train_data = data.clone()
    test_data = data.clone()
    for attr in ['x_species', 'y', 'x_species_phylo', 'x_species_traits_nanmask']:
        train_data[attr] = train_data[attr][train_mask]
        test_data[attr] = test_data[attr][test_mask]

    for data_split, mask in zip([train_data, test_data], [train_mask, test_mask]):
        data_split.edge_index_species, data_split.edge_attr_species = subgraph(mask, data.edge_index_species, data.edge_attr_species, relabel_nodes=True)
        data_split.bip_edge_index, data_split.bip_edge_attr = bipartite_subgraph((torch.ones(data.x_spatial.size(0), dtype=torch.bool), mask), 
                                                                                data.bip_edge_index, data.bip_edge_attr, relabel_nodes=True)
    return train_data, test_data

if __name__ == '__main__':
    norm_transform = NormalizeFeatures()
    dataset = FernDataset(Path('data/Ferns'), transform=norm_transform)
    print(dataset)