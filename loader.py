import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import itertools
import pandas as pd
from Bio import Phylo
import networkx as nx
from node2vec import Node2Vec
from sklearn.cluster import KMeans

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset


class FernDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        traits_mean = pd.read_excel(root / 'Traits.xlsx', sheet_name='Mean').set_index('Species')
        traits_var = pd.read_excel(root / 'Traits.xlsx', sheet_name='Var').drop(columns=['Family', 'Habitat', 'Hybridisation']).set_index('Species')
        traits_std = traits_var.apply(np.sqrt).rename(columns=lambda x: x.replace('Var', 'Std'))

        self.traits_all = traits_mean.join(traits_std)
        self.traits_all = self.traits_all.drop(index='Hymenophyllum_falklandicum') # missing information

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

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
    
    def get_traits_df(self):
        for cl_feat in ['Habitat', 'Family']:
            for cl in self.traits_all[cl_feat].unique():
                if self.traits_all[cl_feat].eq(cl).sum() < len(self.traits_all) * 0.05:
                    self.traits_all[cl_feat] = self.traits_all[cl_feat].replace({cl: 'Other'})
        self.traits_all = pd.get_dummies(self.traits_all, drop_first=True)

        self.class_idx = {
            'Habitat': [self.traits_all.columns.get_loc(c) for c in self.traits_all.columns if c.startswith('Habitat')],
            'Family': [self.traits_all.columns.get_loc(c) for c in self.traits_all.columns if c.startswith('Family')],
        }

    @staticmethod
    def load_raster(f, grid_step=.1):
        raster = xr.open_dataarray(f)
        current_step = (raster.x[1] - raster.x[0]).values # degrees
        downsample = round(grid_step / current_step)
        raster = raster.coarsen(x=downsample, y=downsample, boundary="trim").sum().fillna(0)
        return raster

    def load_complete(self, data_path):
        comp_rasters = []
        for f in tqdm(list((Path(self.root) / "complete layers"/data_path).rglob('*.tif')), desc=f'Loading {data_path}'):
            raster = self.load_raster(f)
            filt_raster = raster.interp(x=self.space_df.x.values, y=self.space_df.y.values, method='nearest').isel(band=0)
            filt_raster = filt_raster.to_pandas().reset_index().melt(id_vars=['y']).rename(columns={'value': f.stem})
            comp_rasters.append(self.space_df.set_index(['x', 'y']).join(filt_raster.set_index(['x', 'y'])).groupby(['cluster']).mean()[f.stem])
        return pd.concat(comp_rasters, axis=1)

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

    def get_spatial_graph(self):
        self.space_df = pd.read_csv(self.root / 'distriubtion_clusters.csv')
        self.space_clustering = KMeans(n_clusters=50, random_state=0).fit(self.space_df[['x', 'y']])
        self.spatial_graph = Data(pos=torch.tensor(self.space_df.groupby('cluster').agg({'x': 'mean', 'y': 'mean'}).values).float())
        
        self.spatial_graph = KNNGraph(k=6)(self.spatial_graph)
        self.spatial_graph.edge_attr = torch.norm(self.spatial_graph.pos[self.spatial_graph.edge_index[0]] - self.spatial_graph.pos[self.spatial_graph.edge_index[1]], dim=1).unsqueeze(1)

    def get_distribution(self, dist_path):
        ar = self.load_raster(dist_path)

        # give a label to any non zero and non nan value and then aggregate on this dimension
        ar = ar.where(ar > 0, 0)
        ar = ar.where(ar.isnull(), 1)

    def process(self):
        self.get_traits_df()
        self.get_species_graph()
        self.get_spatial_graph()
        clim_rasters = self.load_complete('climatic layers')
        population_elevation_rasters = self.load_complete('population density and elevation layer')
        soil_rasters = self.load_complete('Soil NZ layers')

        self.global_data = pd.concat([clim_rasters, population_elevation_rasters, soil_rasters], axis=1)
        data_list = []

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # self.save(data_list, self.processed_paths[0])

if __name__ == '__main__':
    dataset = FernDataset(Path('data/Ferns'))
    print(dataset)