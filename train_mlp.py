
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


import numpy as np
from datasets import build_dataset_mlp
from models import AutoEncoder

from torch.utils.data import DataLoader, TensorDataset
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


class Trainer:
	def __init__(self, model, train_loader, val_loader, device='cuda', **kwargs):
		self.device = device
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.model = model
		self.lr = kwargs.get('lr', 1e-3)
		self.set_optimizer()

	def set_optimizer(self):
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
		self.scheduler = None # torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=False)

	def train_epoch(self):
		self.model.train()
		recloss_num = 0
		recloss_cat = 0
		for x, known_mask in self.train_loader:
			x, known_mask = x.to(self.device), known_mask.to(self.device)
			self.optimizer.zero_grad()
			output = self.model(x)
			
			num_output = output[:, :known_mask.shape[1]]
			num_gt = x[:, :known_mask.shape[1]]
			cat_output = output[:, known_mask.shape[1]:]
			cat_gt = x[:, known_mask.shape[1]:]
			num_loss = F.mse_loss(num_output[known_mask], num_gt[known_mask])
			cat_loss = F.binary_cross_entropy_with_logits(cat_output, cat_gt)
			recloss = num_loss + cat_loss
			recloss.backward()

			recloss_num += num_loss.item()
			recloss_cat += cat_loss.item()
			self.optimizer.step()

		recloss_num /= len(self.train_loader)
		recloss_cat /= len(self.train_loader)
		recloss_all = recloss_num + recloss_cat
		
		if self.scheduler is not None:
			self.scheduler.step(recloss_all)
		return recloss_num, recloss_cat

	@torch.no_grad()
	def test_epoch(self):
		self.model.eval()
		num_error = 0
		cat_accuracy = 0
		for x, known_mask in self.val_loader:
			x, known_mask = x.to(self.device), known_mask.to(self.device)
			output = self.model(x)
			num_output = output[:, :known_mask.shape[1]]
			num_gt = x[:, :known_mask.shape[1]]
			num_error += F.mse_loss(num_output[known_mask], num_gt[known_mask]).item()

			cat_output = output[:, known_mask.shape[1]:]
			cat_gt = x[:, known_mask.shape[1]:]
			cat_accuracy += ((cat_output > 0) == cat_gt).float().mean().item()
		num_error /= len(self.val_loader)
		cat_accuracy /= len(self.val_loader)
		
		return num_error, cat_accuracy


# -----------------
def main(**kwargs):
	device = torch.device('cuda')
	seed = 42
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	logger = CSVLogger('log.csv')
	X, known_mask, train_idx, val_idx = build_dataset_mlp()

	train_dataset = TensorDataset(torch.tensor(X[train_idx]).float(), torch.tensor(known_mask[train_idx]).bool())
	train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

	val_dataset = TensorDataset(torch.tensor(X[val_idx]).float(), torch.tensor(known_mask[val_idx]).bool())
	val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

	model = AutoEncoder(X.shape[1], [10, 15], 64, dropout=0.0).cuda()

	trainer = Trainer(model, train_loader, val_loader, device=device, **kwargs)
	with trange(kwargs.get('epochs', 1000), desc='Training', unit=' epoch') as bar:
		for epoch in bar:
			results = {}
			results['train_regloss'], results['train_classloss'] = trainer.train_epoch()
			results['val_regerror'], results['val_classacc'] = trainer.test_epoch()
			bar.set_postfix(loss=results['train_regloss'], val_error=results['val_regerror'])
			results['lr'] = trainer.optimizer.param_groups[0]['lr']
			logger.log(results, epoch=epoch)

	logger.finish()
	return model


if __name__ == '__main__':
	main(lr=1e-4, epochs=600)