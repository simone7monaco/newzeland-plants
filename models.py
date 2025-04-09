import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from loader import data_split
from torch_geometric.utils import subgraph, bipartite_subgraph


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=1, num_layers=2, dropout=0.3):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, edge_dim=edge_dim, dropout=dropout)])
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, edge_dim=edge_dim, dropout=dropout))
        self.convs.append(GATConv(hidden_channels, out_channels, edge_dim=edge_dim, dropout=dropout))

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = x.relu()
        return x
    
class TraitsPredictor(nn.Module):
    def __init__(self, in_traits, in_phylo, in_space, hidden_channels, out_channels, num_layers, dropout=0.3):
        super(TraitsPredictor, self).__init__()
        self.space_gnn = GNN(in_space+2, hidden_channels, hidden_channels, num_layers=num_layers)
        self.bipartite_conv = GATConv((hidden_channels, -1), hidden_channels, edge_dim=1, add_self_loops=False)
        self.species_gnn = GNN(in_traits + in_phylo + hidden_channels, hidden_channels, hidden_channels, num_layers=num_layers)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        space_input = torch.cat([data.x_spatial, data.global_data], dim=1)
        space_embeddings = self.space_gnn(space_input, data.edge_index_spatial, data.edge_attr_spatial).relu()

        space_to_species = self.bipartite_conv((space_embeddings, None), data.bip_edge_index, data.bip_edge_attr,
                                               size=(space_embeddings.size(0), data.x_species.size(0)))
        # space_to_species = self.bipartite_conv((space_embeddings, torch.zeros_like(data.x_species)), data.bip_edge_index, data.bip_edge_attr)
        species_input = torch.cat([space_to_species, data.x_species, data.x_species_phylo], dim=1)
        species_embeddings = self.species_gnn(species_input, data.edge_index_species, data.edge_attr_species).relu()
        return self.fc(species_embeddings)

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
        out = out*~batch.x_species_traits_nanmask
        loss = F.mse_loss(out, batch.y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        out = out*~batch.x_species_traits_nanmask
        loss = F.mse_loss(out, batch.y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)