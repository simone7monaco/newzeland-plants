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
import sys
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import KNNGraph, BaseTransform
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import subgraph, bipartite_subgraph


class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes using min-max scaling."""
    def __init__(self, normalizations=None):
        self.normalizations = normalizations or {
            'spatial_x': 'sincos', # They are latitude and longitude
            'spatial_global_data': 'logz',
            'species_x_mean': 'logz', # always positive and sometimes skewed, but managed elsewhere
            'species_x_gen': 'z',
            'species_x_std': 'z',
            'species_x_phylo': None, # x_phylo is a vector of embeddings, so no normalization
            'spatial_spatial_edge_attr': 'z',
            'species_species_edge_attr': 'z',
            'spatial_species_edge_attr': 'z',
        }
        self.props = {}

    def fit(self, data: Data, eps: float = 1e-6):
        if getattr(data, 'props', False):
            print("Warning: Data already has props, overwriting\n")

        for k_norm in self.normalizations:
            if self.normalizations[k_norm] in ['logz', 'z']:
                # log normalization and/or z normalization
                if self.normalizations[k_norm] == 'logz':
                    data_k_norm = torch.log(data[k_norm] + eps)
                else:
                    data_k_norm = data[k_norm]
                mean = data_k_norm.mean(dim=0)
                std = data_k_norm.std(dim=0)
                std[std < eps] = 1.0 # prevent division by zero
                self.props[k_norm] = {'mean': mean, 'std': std}
            elif self.normalizations[k_norm] in ['sincos', None]:
                pass
            else:
                raise ValueError(f"Unknown normalization: {self.normalizations[k_norm]}")
            
    def transform(self, data: Data, eps: float = 1e-6) -> Data:
        return self.forward(data, eps)

    def fit_transform(self, data: Data, eps: float = 1e-6) -> Data:
        self.fit(data, eps)
        return self.forward(data, eps)

    def forward(self, data: Data, eps: float = 1e-6) -> Data:
        if getattr(data, 'normalized', False):
            raise ValueError("Data is already normalized")

        if not self.props:
            print("Warning: No props found, computing on the fly (this may cause data leakage if done on the whole dataset instead of just the training set)\n")
            self.fit(data, eps)
        
        setattr(data, 'normalized', True)
        for k_norm in self.normalizations:
            if self.normalizations[k_norm] == 'sincos':
                data[k_norm] = data[k_norm] * np.pi / 180
                data[k_norm] = torch.stack([torch.sin(data[k_norm][:, 0]), torch.cos(data[k_norm][:, 0]),
                                          torch.sin(data[k_norm][:, 1]), torch.cos(data[k_norm][:, 1])], dim=1)
            elif self.normalizations[k_norm] in ['logz', 'z']:
                # log normalization and/or z normalization
                if self.normalizations[k_norm] == 'logz':
                    data[k_norm] = torch.log(data[k_norm] + eps)
                
                data[k_norm] = (data[k_norm] - self.props[k_norm]['mean']) / self.props[k_norm]['std']

            elif self.normalizations[k_norm] is None:
                pass
            else:
                raise ValueError(f"Unknown normalization: {self.normalizations[k_norm]}")
            if k_norm in ['species_x_mean', 'species_x_std']:
                # prevent points in "nan_mask" from being normalized (they will be set to 0 after normalization, which is the mean value of the non-nan points)
                data[k_norm] = data[k_norm] * ~data.traits_nanmask
        return data
    
    def inverse(self, data, warn=True):
        if not getattr(data, 'normalized', False):
            raise ValueError("Data is not normalized")
        data.normalized = False

        data = data.clone()
        for k_norm in self.normalizations:
            if k_norm not in data.keys():
                if warn:
                    print(f"Warning: {k_norm} not in data")
                continue
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
    

class NZData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'species_species_edge_index':
            return self.species_num_nodes
        elif key == 'spatial_spatial_edge_index':
            return self.spatial_num_nodes
        elif key == 'spatial_species_edge_index':
            return torch.tensor([[self.spatial_num_nodes], [self.species_num_nodes]])
        return super().__inc__(key, value, *args, **kwargs)

if '__main__' in sys.modules:
    setattr(sys.modules['__main__'], 'NZData', NZData)
# Also add to torch safe globals (weights-only safe loading)
torch.serialization.add_safe_globals([NZData])

class PlantDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.get_traits_df(root)
        self.y_index = np.arange(len(self.traits_mean.columns))

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
    
    @staticmethod
    def load_raster(f, grid_step=.1):
        raster = xr.open_dataarray(f, engine="rasterio")
        current_step = (raster.x[1] - raster.x[0]).values # degrees
        downsample = round(grid_step / current_step)
        raster = raster.coarsen(x=downsample, y=downsample, boundary="trim").sum().fillna(0) # type: ignore
        return raster
    
    
    def get_traits_df(self, root, dummy_threshold=0.05, drop_threshold=0.7):
        def process_trait_col_name(name):
            return name.strip().replace('mean', '').replace('Var', '')
        
        if 'Mean' in pd.ExcelFile(root / 'Traits.xlsx').sheet_names:
            self.traits_mean = pd.read_excel(root / 'Traits.xlsx', sheet_name='Mean').set_index('Species')
            gen_cols = self.traits_mean.select_dtypes(include=['object']).columns.union(['Hybridisation'])
            self.traits_std = pd.read_excel(root / 'Traits.xlsx', sheet_name='Variance').drop(columns=gen_cols).set_index('Species')
            
        else:
            raise RuntimeError('Traits.xlsx file with Mean and Variance sheets not found.')
        self.traits_gen = self.traits_mean[gen_cols]
        self.traits_mean = self.traits_mean.drop(columns=gen_cols)
        self.traits_mean = self.traits_mean.rename(columns=process_trait_col_name)
        
        # convert to St dev
        self.traits_std = self.traits_std.apply(np.sqrt).rename(columns=process_trait_col_name)

        # remove columns with more than drop_threshold missing values
        self.traits_mean = self.traits_mean.drop(columns=self.traits_mean.columns[(self.traits_mean.isna().sum() / len(self.traits_mean)) > drop_threshold])
        for col in self.traits_mean.columns.difference(self.traits_std.columns):
            self.traits_std[col] = np.nan
        self.traits_std = self.traits_std[self.traits_mean.columns]

        for cl_feat in gen_cols:
            for cl in self.traits_gen[cl_feat].unique():
                if self.traits_gen[cl_feat].eq(cl).sum() < len(self.traits_gen) * dummy_threshold and not pd.isna(cl):
                    self.traits_gen[cl_feat] = self.traits_gen[cl_feat].replace({cl: 'Other'})
        self.traits_gen = pd.get_dummies(self.traits_gen, drop_first=True)
        return self.traits_mean, self.traits_std, self.traits_gen

    @property
    def raw_dir(self):
        return Path(self.root) / 'Distribution layers'
    
    @property
    def raw_file_names(self):
        dist_files = [f'{f}_distribution.tif' for f in self.traits_mean.index]
        return dist_files

    @property
    def processed_file_names(self):
        return ['heterodata.pt']

    def download(self):
        raise RuntimeError('Dataset not found.')

    def load_complete(self, data_path):
        comp_rasters = []
        df_path = (Path(self.root) / "Complete layers"/"{data_path}_space_df.csv")
        if df_path.exists():
            return pd.read_csv(df_path, index_col=0)
        
        for f in tqdm(list((Path(self.root) / "Complete layers"/data_path).rglob('*.tif')), desc=f'Loading {data_path}'):
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
        tree_file = next(Path(self.root).glob('*.nwk'), None)
        if tree_file is None:
            raise RuntimeError('Phylogenetic tree file not found.')
        tree = Phylo.read(tree_file, "newick") # pyright: ignore[reportPrivateImportUsage]

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

        G_species.remove_nodes_from(list(set(G_species.nodes).difference(self.traits_mean.index)))

        for node in G_species.nodes():
            G_species.nodes[node]["x"] = node_embeddings[node]
        self.species_graph = from_networkx(G_species)  # Include node attributes
        self.species_graph.node_names = [n for n in G_species.nodes()]  # Map node names to indices

        # Add nodes not present in the ph_tree
        for node in self.traits_mean.index.difference(self.species_graph.node_names):
            self.species_graph.node_names.append(node)
            self.species_graph.x = torch.cat([self.species_graph.x, torch.zeros(1, self.species_graph.num_features)], dim=0)

        self.species_graph.traits_nanmask = torch.tensor(self.traits_mean.loc[self.species_graph.node_names].isna().values, dtype=torch.bool)
        traits_nanmask_std = torch.tensor(self.traits_std.loc[self.species_graph.node_names].isna().values, dtype=torch.bool)
        # assert no non-nan std is nan in mean
        assert torch.all((~traits_nanmask_std) <= (~self.species_graph.traits_nanmask))

        self.traits_mean = self.traits_mean.loc[self.species_graph.node_names]
        self.traits_std = self.traits_std.loc[self.species_graph.node_names]
        self.traits_gen = self.traits_gen.loc[self.species_graph.node_names]
        self.species_graph.traits_mean = torch.tensor(self.traits_mean.fillna(0).astype(np.float32).values)
        self.species_graph.traits_std = torch.tensor(self.traits_std.fillna(0).astype(np.float32).values)
        self.species_graph.x_gen = torch.tensor(self.traits_gen.astype(np.float32).values)
        return self.species_graph

    def get_spatial_graph(self, n_clusters=50, k=6):
        all_occurrences = self.load_raster(Path(self.root)/"Distribution layers/_all_species_distributions.tif")
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

        ar = ar.where(ar > 0, 0)
        ar = ar.where(ar.isnull(), 1)

    def process(self):
        # TODO: Ablation studies 
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
        species_x_mean=species_graph.traits_mean,
        species_x_std=species_graph.traits_std,
        species_x_gen=species_graph.x_gen,
        species_names=species_graph.node_names,
        species_x_phylo=species_graph.x,
        traits_nanmask=species_graph.traits_nanmask,
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
        num_nodes=species_graph.num_nodes + spatial_graph.num_nodes,
        )
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
    splits = [s for s in splitter.split(G.nodes())] # type: ignore
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
    for attr in ['x_mean', 'x_std', 'x_gen', 'x_phylo',]:
        train_data[f'species_{attr}'] = train_data[f'species_{attr}'][train_mask]
        test_data[f'species_{attr}'] = test_data[f'species_{attr}'][test_mask]
    train_data.traits_nanmask = train_data.traits_nanmask[train_mask]
    test_data.traits_nanmask = test_data.traits_nanmask[test_mask]
    train_data.species_num_nodes = train_mask.sum().item()
    test_data.species_num_nodes = test_mask.sum().item()
    train_data.num_nodes = train_data.species_num_nodes + train_data.spatial_num_nodes
    test_data.num_nodes = test_data.species_num_nodes + test_data.spatial_num_nodes
    

    for data_split, mask in zip([train_data, test_data], [train_mask, test_mask]):
        data_split.species_species_edge_index, data_split.species_species_edge_attr = subgraph(
            mask, edge_index=data.species_species_edge_index,
            edge_attr=data.species_species_edge_attr,
            relabel_nodes=True
            )
        data_split.spatial_species_edge_index, data_split.spatial_species_edge_attr = bipartite_subgraph( # type: ignore
            (torch.ones(data.spatial_num_nodes, dtype=torch.bool), mask), 
            edge_index=data.spatial_species_edge_index, edge_attr=data.spatial_species_edge_attr, relabel_nodes=True
            )
    return train_data, test_data

if __name__ == '__main__':
    norm_transform = NormalizeFeatures()
    data = PlantDataset(Path('data/Ferns'))[0]
    normed_data = norm_transform(data)
    re_unnormed_data = norm_transform.inverse(normed_data)

    for k in norm_transform.normalizations:
        if not torch.allclose(data[k], re_unnormed_data[k]): # type: ignore
            print(f"Normalization failed for {k} (max diff: {torch.max(torch.abs(data[k] - re_unnormed_data[k]))})")
        # n, k = k.split('/')
        # if len(n.split('-')) > 1:
        #     n = n.split('-')
        # if not torch.allclose(data[n][k], re_unnormed_data[n][k]):
        #     print(f"Normalization failed for {n}: {k} (max diff: {torch.max(torch.abs(data[n][k] - re_unnormed_data[n][k]))})")

    