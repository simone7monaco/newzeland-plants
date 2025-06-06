import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from loader import data_split
from torch_geometric.utils import subgraph, bipartite_subgraph
from torch_geometric.data import Data, HeteroData
import torch_geometric


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn, edge_dim=1, num_layers=2, dropout=0.3, **kwargs):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList([gnn(in_channels, hidden_channels, edge_dim=edge_dim, dropout=dropout, **kwargs)])
        for _ in range(num_layers - 2):
            self.convs.append(gnn(hidden_channels, hidden_channels, edge_dim=edge_dim, dropout=dropout, **kwargs))
        self.convs.append(gnn(hidden_channels, out_channels, edge_dim=edge_dim, dropout=dropout))

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None, **kwargs):
        attention_weight = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr, 
                     return_attention_weights=return_attention_weights,
                     **kwargs)
            if return_attention_weights:
                x, attn_weights = x
                attention_weight.append(attn_weights)
            if i < len(self.convs) - 1:
                x = x.relu()
        if return_attention_weights:
            return x, attention_weight
        return x
    
class TraitsPredictor(nn.Module):
    def __init__(self, in_traits, in_phylo, in_space, hidden_channels, out_channels, num_layers, dropout=0.3, gnn_module='GATConv'):
        super(TraitsPredictor, self).__init__()
        gnn = getattr(torch_geometric.nn, gnn_module)

        pos_embedding_dim = 4 # sin-cos embeddings for latitude and longitude
        self.space_gnn = GNN(in_space+pos_embedding_dim, hidden_channels, hidden_channels, num_layers=num_layers, gnn=gnn, dropout=dropout)
        # self.bipartite_conv = GNN((hidden_channels, -1), hidden_channels, hidden_channels, edge_dim=1, num_layers=num_layers-1, add_self_loops=False)
        self.bipartite_conv = GATConv((hidden_channels, -1), hidden_channels, edge_dim=1, add_self_loops=False)
        self.species_gnn = GNN(in_traits + in_phylo + hidden_channels, hidden_channels, hidden_channels, num_layers=num_layers, gnn=gnn, dropout=dropout)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.space_attention_weights = None
        self.bip_attention_weights = None
        self.species_attention_weights = None
    
    def forward(self, data: HeteroData, return_attention_weights=None):
        space_input = torch.cat(
            [data.spatial_x, data.spatial_global_data], 
            dim=1) if hasattr(data, 'spatial_global_data') else data.spatial_x
        species_input = torch.cat(
            [data.species_x, data.species_x_phylo], dim=1) if hasattr(data, 'species_x_phylo') else data.species_x
        
        space_embeddings = self.space_gnn(space_input, data.spatial_spatial_edge_index,
                                          edge_attr=data.spatial_spatial_edge_attr,
                                          return_attention_weights=return_attention_weights)
        if return_attention_weights:
            space_embeddings, attention_weights = space_embeddings
            self.space_attention_weights = attention_weights
        space_embeddings = space_embeddings.relu()

        space_to_species = self.bipartite_conv((space_embeddings, None), data.spatial_species_edge_index,
                                               edge_attr=data.spatial_species_edge_attr,
                                               size=(space_embeddings.size(0), data.species_x.size(0)),
                                               return_attention_weights=return_attention_weights)
        if return_attention_weights:
            space_to_species, bip_attention_weights = space_to_species
            self.bip_attention_weights = bip_attention_weights
        space_to_species = space_to_species.relu()

        species_input = torch.cat([space_to_species, species_input], dim=1)
        species_embeddings = self.species_gnn(species_input, data.species_species_edge_index,
                                              edge_attr=data.species_species_edge_attr,
                                              return_attention_weights=return_attention_weights)
        if return_attention_weights:
            species_embeddings, species_attention_weights = species_embeddings
            self.species_attention_weights = species_attention_weights
        species_embeddings = species_embeddings.relu()

        out = self.fc(species_embeddings)
        if self.training:
            out = out * ~data.traits_nanmask
        return out
    
    # def store_spatial(self, data: HeteroData):
    #     """Store the spatial data in the model."""
    #     self.space_data = torch.cat(
    #         [data.spatial_x, data.spatial_global_data], 
    #         dim=1) if hasattr(data['spatial'], 'global_data') else data.spatial_x
    #     self.space_edge_index = data['spatial', 'spatial'].edge_index
    #     self.space_edge_attr = data['spatial', 'spatial'].edge_attr
    #     self.spatial_species_edge_index = data['spatial', 'species'].edge_index
    #     self.spatial_species_edge_attr = data['spatial', 'species'].edge_attr

    # def forward(self, data: Data):
    #     """Forward pass through the model assuming a homogeneous graph on the species."""
    #     assert hasattr(self, 'space_data'), "Spatial data not stored. Call store_spatial() first."

    #     space_embeddings = self.space_gnn(self.space_data, self.space_edge_index, self.space_edge_attr).relu()
    #     species_embeddings = self.species_gnn(data.x, data.edge_index, data.edge_attr).relu()
    #     space_to_species = self.bipartite_conv((space_embeddings, None), self.spatial_species_edge_index,
    #                                            edge_attr=self.spatial_species_edge_attr,
    #                                            size=(space_embeddings.size(0), data.x.size(0)))
    #     species_embeddings = torch.cat([space_to_species, species_embeddings], dim=1)
    #     out = self.fc(species_embeddings)
    #     out = out * ~data.x_phylo[:, 0].isnan()
    #     return out
        

class FernModel(pl.LightningModule):
    def __init__(self, data, lr=0.01):
        super(FernModel, self).__init__()
        self.model = TraitsPredictor(in_traits=data.x_species.size(1), in_phylo=data.x_species_phylo.size(1), 
                        in_space=data.global_data.size(1), hidden_channels=16, out_channels=data.y.size(1),
                        num_layers=2)
        self.data = data
        self.lr = lr

    def setup(self, stage):
        self.data = data_split(self.data)

        self.train_data = self.data.clone()
        self.test_data = self.data.clone()
        for attr in ['x_species', 'y', 'x_species_phylo', 'x_species_traits_nanmask']:
            self.train_data[attr] = self.train_data[attr][self.data.train_mask]
            self.test_data[attr] = self.test_data[attr][self.data.test_mask]

        for ds, mask in zip([self.train_data, self.test_data], [self.data.train_mask, self.data.test_mask]):
            ds.edge_index_species, ds.edge_attr_species = subgraph(mask, self.data.edge_index_species, self.data.edge_attr_species, relabel_nodes=True)
            ds.bip_edge_index, ds.bip_edge_attr = bipartite_subgraph((torch.ones(self.data.x_spatial.size(0), dtype=torch.bool, device=self.data.x_spatial.device), mask), 
                                                                                    self.data.bip_edge_index, self.data.bip_edge_attr, relabel_nodes=True)

    def forward(self, data):
        return self.model(data)
    
    def train_dataloader(self):
        return [self.train_data]
    
    def val_dataloader(self):
        return [self.test_data]

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = F.mse_loss(out, batch.y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = F.mse_loss(out, batch.y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)