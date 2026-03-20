import torch
import numpy as np
from torch import dist, dtype, nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from loader import data_split
from torch_geometric.utils import subgraph, bipartite_subgraph
from torch_geometric.data import HeteroData
import torch_geometric
from torch_geometric import nn as pyg_nn


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn, 
                 edge_dim=1, num_layers=2, dropout=0.3, 
                 check_oversmoothing=True,
                 **kwargs):
        super(GNN, self).__init__()
        if gnn in [pyg_nn.GATConv, pyg_nn.GATv2Conv, pyg_nn.TransformerConv]:
            kwargs['dropout'] = dropout
            kwargs['heads'] = kwargs.get('heads', 4)
            kwargs['concat'] = kwargs.get('concat', False) 
        self.convs = nn.ModuleList([gnn(in_channels, hidden_channels, edge_dim=edge_dim, **kwargs)])
        for _ in range(num_layers - 2):
            self.convs.append(gnn(hidden_channels, hidden_channels, edge_dim=edge_dim, **kwargs))
        self.convs.append(gnn(hidden_channels, out_channels, edge_dim=edge_dim, **kwargs))
        self.check_oversmoothing = check_oversmoothing
        self.dirichlet_energy = []

    @torch.no_grad()
    def log_dirichlet_energy(self, x, edge_index):
        row, cols = edge_index
        diff = x[row] - x[cols]
        e = diff.pow(2).sum(dim=-1).mean()
        self.dirichlet_energy.append(e.item())
        return e.item()
        
    def forward(self, x, edge_index, edge_attr=None, 
                return_attention_weights=None, **kwargs):
        attention_weight = []
        for i, conv in enumerate(self.convs):
            if self.check_oversmoothing:
                # sanity check for oversmoothing
                e = self.log_dirichlet_energy(x, edge_index)
                if e < 1e-5:
                    print(f'Warning: Low Dirichlet energy {e} at layer {i}, possible oversmoothing.')
            
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
    def __init__(self, in_traits, in_gen, in_phylo, in_space, hidden_channels, out_channels, 
                 num_layers, dropout=0.3, gnn_module='GATConv', eps=1e-6, mask_ratio=0.15, 
                 mask_strategy='random', use_env_features=True):
        super(TraitsPredictor, self).__init__()
        gnn = getattr(pyg_nn, gnn_module)

        if use_env_features:
            pos_embedding_dim = 4 # sin-cos embeddings for latitude and longitude
            self.space_gnn = GNN(in_space+pos_embedding_dim, hidden_channels, hidden_channels, num_layers=num_layers, gnn=gnn, dropout=dropout)
            
            # Species-side: considering using only non-target  inputs at that stage
            self.bipartite_conv = pyg_nn.GATConv((hidden_channels, in_gen + in_phylo), hidden_channels, edge_dim=1, add_self_loops=False)

        # mean_traits, std_traits, visibility_mask, gen_features, phylo_features
        self.species_linear = nn.Linear(3*in_traits + in_gen + in_phylo, hidden_channels)
        self.species_gnn = GNN(hidden_channels + hidden_channels, hidden_channels, hidden_channels, num_layers=num_layers, gnn=gnn, dropout=dropout)
        
        # Predict mean and log_std separately (2 * out_channels)
        self.fc = nn.Linear(hidden_channels, 2 * out_channels)
        self.eps = eps  # Small constant for numerical stability
        self.mask_ratio = mask_ratio  # Fraction of observed traits to mask during training
        self.mask_strategy = mask_strategy  # 'random', 'blockwise', or 'balanced'
        self.use_env_features = use_env_features
        self.space_attention_weights = None
        self.bip_attention_weights = None
        self.species_attention_weights = None
        self.reconstruction_mask = None

    def compute_reconstruction_mask(self, species_mean, observed_mask):
        reconstruction_mask = None
        if self.training and self.mask_ratio > 0:
            # Identify observed (non-NaN) trait entries
            
            if self.mask_strategy == 'blockwise':
                # Mask entire traits (columns) to force cross-trait learning
                n_traits = species_mean.size(1)
                n_traits_to_mask = max(1, int(n_traits * self.mask_ratio))
                traits_to_mask = torch.randperm(n_traits)[:n_traits_to_mask].to(species_mean.device)
                reconstruction_mask = torch.zeros_like(species_mean, dtype=torch.bool)
                reconstruction_mask[:, traits_to_mask] = True
                reconstruction_mask = reconstruction_mask & observed_mask
            elif self.mask_strategy == 'balanced':
                # Ensure each trait is masked with similar frequency
                reconstruction_mask = torch.zeros_like(species_mean, dtype=torch.bool)
                for trait_idx in range(species_mean.size(1)):
                    trait_observed = observed_mask[:, trait_idx]
                    n_observed = trait_observed.sum()
                    if n_observed > 0:
                        n_to_mask = max(1, int(n_observed * self.mask_ratio))
                        observed_indices = torch.where(trait_observed)[0]
                        mask_indices = observed_indices[torch.randperm(len(observed_indices))[:n_to_mask].to(species_mean.device)]
                        reconstruction_mask[mask_indices, trait_idx] = True
            else:  # 'random' (default)
                # Randomly select observed entries to mask for reconstruction
                reconstruction_mask = torch.rand_like(species_mean) < self.mask_ratio
                reconstruction_mask = reconstruction_mask & observed_mask  # Only mask observed entries
        return reconstruction_mask
        
    def forward(self, data: HeteroData, return_attention_weights=None):
        """
        Forward pass with optional masked reconstruction during training.
        
        During training, randomly masks a fraction (mask_ratio) of observed traits in the input,
        forcing the model to predict them from graph structure, phylogeny, and geography.
        This prevents shortcut learning where the model just copies input traits to output.
        
        Returns:
            If training: (pred_mean, pred_std, reconstruction_mask)
                reconstruction_mask indicates which traits were masked and should be used for loss
            If not training: (pred_mean, pred_std)
        """
        
        # Create a copy of traits that may be masked during training
        species_x_mean = data.species_x_mean.clone()
        species_x_std = data.species_x_std.clone()

        observed_mask = ~data.traits_nanmask  # Shape: (n_species, n_traits)
        reconstruction_mask = self.compute_reconstruction_mask(data.species_x_mean, observed_mask)
        if reconstruction_mask is not None:
            # Zero out masked entries in the input (sentinel value)
            species_x_mean = species_x_mean * ~reconstruction_mask
            species_x_std = species_x_std * ~reconstruction_mask
            # Binary indicator: 1 where the model can see the true value, 0 where missing or masked
            visibility_mask = (observed_mask & ~reconstruction_mask).float()
        else:
            visibility_mask = observed_mask.float()
        
        species_input = torch.cat([species_x_mean, species_x_std, visibility_mask, data.species_x_gen, data.species_x_phylo], dim=1)
        species_input = self.species_linear(species_input).relu()
        
        if self.use_env_features:
            space_input = torch.cat([data.spatial_x, data.spatial_global_data], dim=1)
            space_embeddings = self.space_gnn(space_input, data.spatial_spatial_edge_index,
                                            edge_attr=data.spatial_spatial_edge_attr,
                                            return_attention_weights=return_attention_weights)
            if return_attention_weights:
                space_embeddings, attention_weights = space_embeddings
                self.space_attention_weights = attention_weights
            space_embeddings = space_embeddings.relu()

            species_part = torch.cat([data.species_x_gen, data.species_x_phylo], dim=1)
            space_to_species = self.bipartite_conv((space_embeddings, species_part), data.spatial_species_edge_index,
                                edge_attr=data.spatial_species_edge_attr,
                                size=(space_embeddings.size(0), data.species_x_mean.size(0)),
                                return_attention_weights=return_attention_weights)
            if return_attention_weights:
                space_to_species, bip_attention_weights = space_to_species
                self.bip_attention_weights = bip_attention_weights
            space_to_species = space_to_species.relu()
            # space_to_species = torch.zeros_like(space_to_species)

            species_input = torch.cat([space_to_species, species_input], dim=1)

        species_embeddings = self.species_gnn(species_input, data.species_species_edge_index,
                                              edge_attr=data.species_species_edge_attr,
                                              return_attention_weights=return_attention_weights)
        if return_attention_weights:
            species_embeddings, species_attention_weights = species_embeddings
            self.species_attention_weights = species_attention_weights
        species_embeddings = species_embeddings.relu()

        # Predict mean and log_std
        mean_std = self.fc(species_embeddings)
        mean, log_std = mean_std.chunk(2, dim=-1)
        
        # Apply softplus to log_std and add epsilon: sigma = softplus(log_std) + eps
        std = F.softplus(log_std) + self.eps
             
        self.reconstruction_mask = reconstruction_mask
        return mean, std

    @staticmethod
    def compute_coverage(pred_mean, pred_std, target_mean, confidence_levels=[0.9, 0.95], nanmask=None):
        """
        Compute coverage: fraction of targets within confidence intervals.
        
        Args:
            pred_mean: Predicted means (batch, features)
            pred_std: Predicted standard deviations (batch, features)
            target_mean: Target values (batch, features)
            confidence_levels: List of confidence levels to evaluate (e.g., [0.9, 0.95])
            nanmask: Boolean mask indicating missing values (batch, features)
        
        Returns:
            Dict mapping confidence level to coverage fraction
        """
        if nanmask is not None:
            mask = ~nanmask
        else:
            mask = torch.ones_like(pred_mean, dtype=torch.bool)
        
        from scipy import stats
        coverage = {}
        
        for conf in confidence_levels:
            # Compute z-score for confidence level
            z = stats.norm.ppf((1 + conf) / 2)
            
            # Compute confidence intervals
            lower = pred_mean - z * pred_std
            upper = pred_mean + z * pred_std
            
            # Check if targets fall within intervals
            within_interval = ((target_mean >= lower) & (target_mean <= upper) & mask).float()
            coverage[conf] = within_interval.sum() / mask.sum()
        
        return coverage

class DeterministicLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DeterministicLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_mean, pred_std, target_mean, target_std=None, mask=None) -> torch.Tensor:
        """
            Deterministic loss: Huber loss on both mean and std predictions.
        """

        if mask is None:
            mask = torch.ones_like(pred_mean, dtype=torch.bool)

        n_valid = mask.sum()
        if n_valid == 0:
            raise Warning("No valid entries to compute loss on.")
        
        huber_mean = F.huber_loss(
            pred_mean[mask],
            target_mean[mask],
            reduction='sum',
        )
        # some std targets may be NaN or zero; ignore those in loss
        std_mask = mask & (~torch.isnan(target_std)) & (target_std > 0)
        huber_std = F.huber_loss(
            pred_std[std_mask],
            target_std[std_mask],
            reduction='sum',
        )
        self.cache = {
            'huber_mean': huber_mean.mean().item(),
            'huber_std': huber_std.mean().item(),
        }
        full_loss = huber_mean + huber_std
        
        if self.reduction == "sum":
            return full_loss
        elif self.reduction == "mean":
            return full_loss / n_valid
        else:
            raise ValueError("reduction must be 'mean' or 'sum'.")
        
class MixedNLLLoss(nn.Module):
    def __init__(self, distribution='lognormal', reduction='mean',
                 kl_weight=1.0, eps=1e-6):
        super(MixedNLLLoss, self).__init__()
        self.eps = eps
        self.distribution = distribution
        self.reduction = reduction
        self.kl_weight = kl_weight
        self.cache = {}

    def _lognormal_mean_std_to_normal_params(self, mean, std):
        """
        Convert LogNormal mean/std in original space (m, s) to underlying Normal params (mu, sigma)
        such that log(X) ~ Normal(mu, sigma^2).

        Given:
            m = E[X] > 0
            s = Std[X] >= 0

        Then:
            sigma^2 = log(1 + (s/m)^2)
            mu      = log(m) - 0.5*sigma^2
        """

        log_m = torch.log(mean)
        log_s = torch.log(std)
        log_cv2 = 2 * (log_s - log_m)

        sigma2 = F.softplus(log_cv2)
        mu = log_m - 0.5 * sigma2
        return mu, sigma2.sqrt()
    
    def forward(self, pred_mean, pred_std, target_mean, target_std=None, mask=None) -> torch.Tensor:
        """
            Mixed loss:
            - Point targets: Huber loss on mean
            - Distribution targets: KL divergence between target and predicted distributions
        """

        if mask is None:
            mask = torch.ones_like(pred_mean, dtype=torch.bool)

        n_valid = mask.sum()
        if n_valid == 0:
            raise Warning("No valid entries to compute loss on.")
        
        if target_std is None:
            is_point_estimate = torch.ones_like(pred_mean, dtype=torch.bool)
        else: # std not nan and > 0
            is_point_estimate = torch.isnan(target_std) | (target_std <= 0)
        
        # Create separate masks for point estimates and distributions
        point_mask = mask & is_point_estimate
        dist_mask = mask & ~is_point_estimate

        loss_sum = torch.zeros((), device=pred_mean.device)

        # ---- Point targets: Huber on mean in ORIGINAL space ----
        if point_mask.any():
            huber = F.huber_loss(
                pred_mean[point_mask],
                target_mean[point_mask],
                reduction="sum",
            )
            loss_sum = loss_sum + huber #* point_mask.sum()

        # ---- Distribution targets: KL divergence ----
        if dist_mask.any():
            # Clamp predicted std away from 0 for numerical stability in KL.
            # (We do NOT use pred_std for point targets; only for dist targets.)
            if self.distribution == "normal":
                mu_p = pred_mean[dist_mask]
                sig_p = pred_std[dist_mask].clamp_min(self.eps)

                mu_t = target_mean[dist_mask]
                sig_t = target_std[dist_mask].clamp_min(self.eps)

            else:
                pm = pred_mean[dist_mask].clamp_min(self.eps)
                ps = pred_std[dist_mask].clamp_min(0.0)

                tm = target_mean[dist_mask].clamp_min(self.eps)
                ts = target_std[dist_mask].clamp_min(0.0)

                mu_p, sig_p = self._lognormal_mean_std_to_normal_params(pm, ps)
                mu_t, sig_t = self._lognormal_mean_std_to_normal_params(tm, ts)

                sig_p = sig_p.clamp_min(self.eps)
                sig_t = sig_t.clamp_min(self.eps)

            # KL( N(mu_t, sig_t^2) || N(mu_p, sig_p^2) )
            kl = torch.log(sig_p / sig_t) + (sig_t.pow(2) + (mu_t - mu_p).pow(2)) / (2.0 * sig_p.pow(2)) - 0.5
            # KL( N(mu_p, sig_p^2) || N(mu_t, sig_t^2) )
            # kl = torch.log(sig_t / sig_p) + (sig_p.pow(2) + (mu_p - mu_t).pow(2)) / (2.0 * sig_t.pow(2)) - 0.5

            loss_sum = loss_sum + self.kl_weight * kl.sum() #* dist_mask.sum()

        self.cache = {
            'huber': huber.mean().item() if point_mask.any() else 0.0,
            'kl': kl.mean().item() if dist_mask.any() else 0.0,
        }


        if self.reduction == "sum":
            return loss_sum
        elif self.reduction == "mean":
            return loss_sum / (dist_mask.sum() + point_mask.sum())
        else:
            raise ValueError("reduction must be 'mean' or 'sum'.")




def graph_smoothness_loss(pred_mean, edge_index, nanmask=None, reduction='mean'):
    """
    Graph smoothness regularization: encourages predictions to be smooth over the graph.
    Useful as auxiliary loss on truly missing traits (where no ground truth exists).
    
    Penalizes large differences between connected nodes, encouraging the model to
    produce coherent predictions based on graph structure.
    
    Args:
        pred_mean: Predicted means (batch, features)
        edge_index: Graph edges (2, num_edges)
        nanmask: Boolean mask indicating missing values (batch, features)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Smoothness loss (scalar or per-edge if reduction='none')
    """
    row, col = edge_index
    
    # Compute squared differences between connected nodes
    diff = pred_mean[row] - pred_mean[col]
    smoothness = diff.pow(2)  # Shape: (num_edges, features)
    
    if nanmask is not None:
        # Only penalize smoothness on truly missing traits
        missing_mask = nanmask  # True where traits are missing
        # For each edge, only consider traits that are missing in at least one of the nodes
        edge_missing_mask = missing_mask[row] | missing_mask[col]
        smoothness = smoothness * edge_missing_mask
    
    if reduction == 'mean':
        return smoothness.mean()
    elif reduction == 'sum':
        return smoothness.sum()
    else:
        return smoothness


# class FernModel(pl.LightningModule):
#     def __init__(self, data, lr=0.01):
#         super(FernModel, self).__init__()
#         self.model = TraitsPredictor(in_traits=data.x_species.size(1), in_phylo=data.x_species_phylo.size(1), 
#                         in_space=data.global_data.size(1), hidden_channels=16, out_channels=data.y.size(1),
#                         num_layers=2)
#         self.data = data
#         self.lr = lr

#     def setup(self, stage):
#         self.data = data_split(self.data)

#         self.train_data = self.data.clone()
#         self.test_data = self.data.clone()
#         for attr in ['x_species', 'y', 'x_species_phylo', 'x_species_traits_nanmask']:
#             self.train_data[attr] = self.train_data[attr][self.data.train_mask]
#             self.test_data[attr] = self.test_data[attr][self.data.test_mask]

#         for ds, mask in zip([self.train_data, self.test_data], [self.data.train_mask, self.data.test_mask]):
#             ds.edge_index_species, ds.edge_attr_species = subgraph(mask, self.data.edge_index_species, self.data.edge_attr_species, relabel_nodes=True)
#             ds.bip_edge_index, ds.bip_edge_attr = bipartite_subgraph((torch.ones(self.data.x_spatial.size(0), dtype=torch.bool, device=self.data.x_spatial.device), mask), 
#                                                                                     self.data.bip_edge_index, self.data.bip_edge_attr, relabel_nodes=True)

#     def forward(self, data):
#         return self.model(data)
    
#     def train_dataloader(self):
#         return [self.train_data]
    
#     def val_dataloader(self):
#         return [self.test_data]

#     def training_step(self, batch, batch_idx):
#         out = self.model(batch)
#         loss = F.mse_loss(out, batch.y)
#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         out = self.model(batch)
#         loss = F.mse_loss(out, batch.y)
#         self.log('val_loss', loss)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)