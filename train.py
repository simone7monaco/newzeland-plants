import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from loader import FernDataset, NormalizeFeatures, data_split
from models import TraitsPredictor
from tqdm import trange
from pathlib import Path
import pytorch_lightning as pl
from copy import deepcopy
import wandb


torch.set_float32_matmul_precision('medium')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
pl.seed_everything(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('-e', '--epochs', type=int, default=300, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--gnn_module', type=str, default='GATv2Conv', help="GNN attention module", choices=['GATConv', 'Gatv2Conv', 'TransformerConv'])
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels in the GNN')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the GNN')
    parser.add_argument('--test_logging_step', type=int, default=1, help='Step for test logging')
    parser.add_argument('--save_model', action='store_true', help='Save the model after training')
    return parser.parse_args()

def main(args):
    wandb.init(project='fern-sweep', config=args, mode='disabled')
    for key, value in wandb.config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    print(f"---------------\nTraining with args: {args}")

    norm_transform = NormalizeFeatures()
    dataset = FernDataset(Path('data/Ferns'), transform=norm_transform)
    data = dataset[0]

    model = TraitsPredictor(in_traits=data.species_x.size(1), in_phylo=data.species_x_phylo.size(1), 
                            in_space=data.spatial_global_data.size(1), out_channels=data.species_y.size(1), 
                            hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                            dropout=args.dropout, gnn_module=args.gnn_module)

    train_data, test_data = data_split(data, k=args.k, seed=seed)
        
    model = model.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    loss_fn = torch.nn.MSELoss()
    Loss = []
    test_losses = []
    best_test_loss = float('inf')
    best_model = None
    best_epoch = 0
    patience = 0

    for epoch in trange(args.epochs, desc="Training", unit="epoch"):
        model.train()
        optimizer.zero_grad()
        out = model(train_data)
        loss = loss_fn(out, train_data.species_y)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        wandb.log({'train_loss': loss.item(), 'epoch': epoch})
        Loss.append(loss.item())
        
        if epoch % args.test_logging_step == 0:
            model.eval()
            with torch.no_grad():
                out = model(test_data)
                test_loss = loss_fn(out, test_data.species_y)
                test_losses.append(test_loss.item())
                if test_loss < best_test_loss:
                    best_test_loss = test_loss.item()
                    best_epoch = epoch
                    best_model = deepcopy(model.state_dict())
                # elif test_loss.item() - best_test_loss > 1e-2:
                #     patience += 1
                #     if patience >= 10:
                #         print(f"Early stopping at epoch {epoch} with patience {patience}")
                #         break
                wandb.log({'test_loss': test_loss.item(), 
                           'best_test_loss': best_test_loss,
                           'epoch': epoch})

        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.plot(Loss, label='Train Loss')
        ax.plot(range(0, epoch+1, args.test_logging_step), test_losses, label='Test Loss')
        ax.scatter(best_epoch, best_test_loss, color='red', label='Best Test Loss', marker='*')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        ax.legend()

        ax.set_yscale('log')
        fig.savefig(f'train_{k}.png')
        plt.close(fig)

    # Save the model
    if args.save_model:
        torch.save(best_model, f'best_model_{k}.pth')
    print(f"Best test loss: {best_test_loss} at epoch {best_epoch}")

if __name__ == "__main__":
    args = get_args()
    for k in range(5):
        print(f"Running fold {k+1}/5")
        args.k = 4-k
        main(args)
        


# sweep_id = wandb.sweep(sweep_config, project='fern-sweep')
# wandb.agent(sweep_id, lambda: main(args))
# sweep_config = {
#     'name': 'fern-sweep',
#     'method': 'grid',
#     'metric': {
#         'name': 'best_test_loss',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'epochs': {'values': [1000]},
#         'lr': {'values': [0.0005, 0.001, 0.005]},
#         'hidden_channels': {'values': [16, 32, 64]},
#         'num_layers': {'values': [2, 3, 4]},
#         'dropout': {'values': [0.1, 0.2, 0.3]},
#         'gnn_module': {'values': ['GATConv', 'Gatv2Conv', 'TransformerConv']}
#     }
# }