
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


import numpy as np
from datasets import SpecieDataset, build_dataset
from models import GCNEncoder, GCNDecoder, NodeVGAE

from torch_geometric.loader import DataLoader
from tqdm.auto import trange


class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.DataFrame()
        # put a placeholder
        self.df.to_csv(self.filename, index=False)
        self.best_mse = 1e10

    def log(self, results: dict, **kwargs):
        results = pd.DataFrame([results])
        for k, v in kwargs.items():
            results[k] = v
        self.df = pd.concat([self.df, results], ignore_index=True)

    def flush(self):
        self.df.to_csv(self.filename, index=False)

    def update_best(self, mse):
        if mse < self.best_mse:
            self.best_mse = mse

    def finish(self):
        self.df.to_csv(self.filename, index=False)


	

class TraitLoss(nn.Module):
	def __init__(self, weight=None, reduction='mean'):
		super().__init__()
		"""
		weight (optional): tensor of shape (trait_classes, 1) to weight the loss of each trait class
		"""
		self.weight = weight
		self.reduction = reduction
		self.regr_loss_fn = nn.MSELoss(reduction='none')
		self.class_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

	def forward(self, input, pred, target):
		"""
		input is a tensor (batch_size * n_traits, 3), pred and target are of shape (batch_size * n_traits, 1). The loss is a MSE on the conditioned to:
		- a +1 on the second column of input means "absent": the ground truth is -1 (already in target)
		- a -1 on the third column (with second column != 1) of input means "present but unknown": don't consider the loss
		- other possibilities means "present": the ground truth is the value on the third column (or in pred as well)
		"""
		unknown_mask = input[:, 1].ne(1) & input[:, 2].eq(-1)
		regr_loss = self.regr_loss_fn(pred[~unknown_mask&input[:, 0].ne(10)], target[~unknown_mask&input[:, 0].ne(10)])
		class_loss = self.class_loss_fn(pred[~unknown_mask&input[:, 0].eq(10)], target[~unknown_mask&input[:, 0].eq(10)])
		loss = torch.cat([regr_loss, class_loss], dim=0)

		if self.weight is not None:
			loss = loss * self.weight
		if self.reduction in ['mean', 'sum']:
			return getattr(loss, self.reduction)()
		return loss

class Trainer:
	def __init__(self, model, train_loader, val_loader, device='cuda', **kwargs):
		self.device = device
		self.reconstruction_loss = TraitLoss()
		self.kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		# self.logger = logger
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.model = model
		self.lr = kwargs.get('lr', 1e-5)
		self.set_optimizer(model)

	def set_optimizer(self, model):
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)

	def train_epoch(self):
		self.model.train()
		recloss_all = 0
		klloss_all = 0
		for data in self.train_loader:
			data = data.to(self.device)
			self.optimizer.zero_grad()
			output, mu, logvar = self.model(data.x, data.edge_index)

			rec_loss = self.reconstruction_loss(data.x, output, data.y)
			if self.model.use_variation:
				kl_loss = self.kl_loss(mu, logvar)
				loss = rec_loss + kl_loss
				klloss_all += kl_loss.item() * data.num_graphs
			else:
				loss = rec_loss
			loss.backward()
			recloss_all += rec_loss.item() * data.num_graphs
			self.optimizer.step()
		
		recloss_all /= len(self.train_loader.dataset)

		self.scheduler.step(recloss_all)
		if self.model.use_variation:
			klloss_all /= len(self.train_loader.dataset)
			return recloss_all, klloss_all
		return recloss_all, None


	def test_epoch(self):
		self.model.eval()
		error = 0
		with torch.no_grad():
			for data in self.val_loader:
				data = data.to(self.device)
				output, _, _ = self.model(data.x, data.edge_index)
				# validate only on numerical features (all the ones with no feat id = 10)
				unknown_mask = data.x[:, 1].ne(1) & data.x[:, 2].eq(-1)
				y = data.y[~unknown_mask & data.x[:, 0].ne(10)]
				y_pred = output[~unknown_mask & data.x[:, 0].ne(10)]
				
				error += F.mse_loss(y_pred, y).item() * data.num_graphs
		
		error = error / len(self.val_loader.dataset)
		
		return error


# -----------------
def main(use_variation=True, **kwargs):
	device = torch.device('cuda')
	seed = 42
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	logger = CSVLogger('log.csv')
	dataset, edges_gen, genus, train_idx, val_idx = build_dataset()

	train_dataset = SpecieDataset('train', indices=train_idx, dataset=dataset, edges_gen=edges_gen, genus=genus, add_self_loops=True)
	train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

	val_dataset = SpecieDataset('val', indices=val_idx, dataset=dataset, edges_gen=edges_gen, genus=genus, add_self_loops=True)
	val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)

	model = NodeVGAE(GCNEncoder(3, 32, 16), GCNDecoder(16, 32, 1), use_variation=use_variation).to(device)

	trainer = Trainer(model, train_loader, val_loader, device=device, **kwargs)
	with trange(kwargs.get('epochs', 250), desc='Training', unit=' epoch') as bar:
		for epoch in bar:
			results = {}
			results['train_recloss'], results['train_klloss'] = trainer.train_epoch()
			results['val_error'] = trainer.test_epoch()
			bar.set_postfix(loss=results['train_recloss'], val_error=results['val_error'])
			results['lr'] = trainer.optimizer.param_groups[0]['lr']
			logger.log(results, epoch=epoch)

	logger.finish()
	return model


if __name__ == '__main__':
	main(False, lr=1e-4, epochs=300)