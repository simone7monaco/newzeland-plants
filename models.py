import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models.autoencoder import VGAE

class EncDecoder(nn.Module):
	def __init__(self, in_channels, hidden_channels:list, latent_channels, activation=nn.LeakyReLU(), dropout=0.):
		super(EncDecoder, self).__init__()
		layers = []
		for i, h in enumerate(hidden_channels):
			layers.append(nn.Linear(in_channels if i == 0 else hidden_channels[i-1], h))
			layers.append(activation)
			layers.append(nn.Dropout(dropout))
		layers.append(nn.Linear(hidden_channels[-1], latent_channels))
		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
	

class AutoEncoder(nn.Module):
	def __init__(self, in_channels, hidden_channels:list, latent_channels, activation=nn.LeakyReLU(), dropout=0.):
		super(AutoEncoder, self).__init__()
		self.encoder = EncDecoder(in_channels, hidden_channels, latent_channels, activation, dropout=dropout)
		self.decoder = EncDecoder(latent_channels, hidden_channels[::-1], in_channels, activation, dropout=dropout)

	def forward(self, x):
		z = self.encoder(x)
		return self.decoder(z)
		
# ----------------- GCN -----------------

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, activation=nn.LeakyReLU()):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, latent_channels)
        self.gcn_logvar = GCNConv(hidden_channels, latent_channels)
        self.activation = activation

    def forward(self, x, edge_index):
        x = self.gcn_shared(x, edge_index)
        x = self.activation(x)
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index).clamp(max=10)
        return mu, logvar
    
class GCNDecoder(nn.Module):
	def __init__(self, latent_channels, hidden_channels, out_channels, activation=nn.LeakyReLU()):
		super(GCNDecoder, self).__init__()
		self.gcn_shared = GCNConv(latent_channels, hidden_channels)
		self.gcn_out = GCNConv(hidden_channels, out_channels)
		self.activation = activation

	def forward(self, z, edge_index):
		z = self.gcn_shared(z, edge_index)
		z = self.activation(z)
		z = self.gcn_out(z, edge_index)
		return z
      
class NodeVGAE(VGAE):
	def __init__(self, encoder, decoder, use_variation=True):
		super(NodeVGAE, self).__init__(encoder, decoder)
		self.encoder = encoder
		self.decoder = decoder
		self.use_variation = use_variation
		
	def reparametrize(self, mu, logvar):
		if self.training:
			std = (0.5 * logvar).exp()
			eps = torch.randn_like(std)
			return eps * std + mu
		else:
			return mu
	
	def forward(self, x, edge_index):
		mu, logvar = self.encoder(x, edge_index)
		if self.use_variation:
			z = self.reparametrize(mu, logvar)
			z = self.decoder(z, edge_index)
		else:
			z = self.decoder(mu, edge_index)
		return z, mu, logvar
