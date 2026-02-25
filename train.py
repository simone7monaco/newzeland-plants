import argparse
import torch
import torch.nn.functional as F
import numpy as np
from loader import PlantDataset, NormalizeFeatures, data_split
from models import TraitsPredictor, DeterministicLoss, MixedNLLLoss, graph_smoothness_loss
from train_baseline import compute_correlation
from tqdm import trange
from pathlib import Path
import pytorch_lightning as pl
from copy import deepcopy
import wandb

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

torch.set_float32_matmul_precision('medium')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
pl.seed_everything(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('-e', '--epochs', type=int, default=300, help='Number of epochs to train the model')
    parser.add_argument('--use_env_features', type=str2bool, default=True, help='Whether to use environmental features')

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--gnn_module', type=str, default='GATv2Conv', help="GNN attention module")#, choices=['GATConv', 'GATv2Conv', 'TransformerConv'])
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in the GNN')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for the GNN')

    parser.add_argument('--loss', type=str, default='dist_normal', choices=['deterministic', 'dist_normal', 'dist_lognormal'], help='Output distribution type')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Masking ratio for input features at training time')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='Weight for the KL divergence loss term')
    parser.add_argument('--smoothness_weight', type=float, default=0.0, help='Weight for the graph smoothness loss')
    
    parser.add_argument('--test_logging_step', type=int, default=1, help='Step for test logging')
    parser.add_argument('--save_model', action='store_true', help='Save the model after training')
    parser.add_argument('--plot', action='store_true', help='Plot training curves')
    parser.add_argument('--use_wb', type=str2bool, default=False, nargs='?', const=True, help='Use Weights & Biases for logging')
    return parser.parse_args()


def main(args):
    wandb.init(project='fern-sweep', config=args, mode='disabled' if not args.use_wb else 'online')
    
    print(f"---------------\nTraining with args: {args}")

    norm_transform = NormalizeFeatures()
    data_path = Path(f'data/Ferns/')
    dataset = PlantDataset(data_path, transform=norm_transform)
    data = dataset[0]

    model = TraitsPredictor(in_traits=data.species_x_mean.size(1), in_gen=data.species_x_gen.size(1), in_phylo=data.species_x_phylo.size(1), 
                            in_space=data.spatial_global_data.size(1), out_channels=data.species_x_mean.size(1), 
                            hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                            dropout=args.dropout, gnn_module=args.gnn_module, use_env_features=args.use_env_features,
                            mask_ratio=args.mask_ratio)

    train_data, test_data = data_split(data, k=args.k, seed=seed)    
       
    model = model.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    if args.loss == 'deterministic':
        loss_fn = DeterministicLoss()
        if args.kl_weight != 0.0:
            print("Warning: KL weight is ignored when using deterministic loss.\n")
    else:
        loss_fn = MixedNLLLoss(distribution=args.loss.split('_')[1], 
                               kl_weight=args.kl_weight)
    
    best_test_loss = float('inf')
    best_model = None
    best_epoch = 0

    for epoch in trange(args.epochs, desc="Training", unit="epoch"):
        model.train()
        optimizer.zero_grad()
        
        pred_mean, pred_std = model(train_data)

         # TODO: put this in the training stage
        if model.reconstruction_mask is not None:
            mask = model.reconstruction_mask
        else:
            mask = ~train_data.traits_nanmask
        loss = loss_fn(pred_mean, pred_std, train_data.species_x_mean, train_data.species_x_std, mask)

        # add graph smothness loss
        gs_loss = graph_smoothness_loss(pred_mean, train_data.species_species_edge_index) * args.smoothness_weight
        loss += gs_loss
        
        if torch.isnan(loss):
            raise ValueError("Loss is NaN. Training stopped.")

        loss.backward()

        optimizer.step()
        scheduler.step(loss.item())
        log_dict = {
            'train_loss': loss.item(), 
            'graph_smoothness_loss': gs_loss.item(),
            } | loss_fn.cache
        
        wandb.log(log_dict, step=epoch)
        
        if epoch % args.test_logging_step == 0:
            model.eval()
            with torch.no_grad():
                pred_mean, pred_std = model(test_data)
                test_loss = loss_fn(pred_mean, pred_std, test_data.species_x_mean, test_data.species_x_std, ~test_data.traits_nanmask)
                
                # Compute additional metrics
                # coverage = model.compute_coverage(
                #     pred_mean, pred_std, test_data.species_x_mean,
                #     confidence_levels=[0.9, 0.95], nanmask=test_data.traits_nanmask
                # )

                # mean RMSE over masked entries (matching baseline)
                mask_for_loss = ~test_data.traits_nanmask
                mean_rmse = F.mse_loss(pred_mean[mask_for_loss], test_data.species_x_mean[mask_for_loss], reduction='mean').sqrt()
                

                correlation = compute_correlation(pred_mean, test_data.species_x_mean)

                if test_loss.item() < best_test_loss:
                    best_test_loss = test_loss.item()
                    best_mean_rmse = mean_rmse.item() if not torch.isnan(mean_rmse) else float('nan')
                    best_correlation = float(correlation)
                    best_epoch = epoch
                    best_model = deepcopy(model.state_dict())
                
                log_dict = {
                    'test_loss': test_loss.item(),
                    'test_mean_rmse': mean_rmse.item() if not torch.isnan(mean_rmse) else float('nan'),
                    'test_correlation': float(correlation),
                    # 'test_coverage_90': coverage[0.9].item(),
                    # 'test_coverage_95': coverage[0.95].item(),
                } | loss_fn.cache
                wandb.log(log_dict, step=epoch)

    # Save the model
    wandb.log({'best_test_loss': best_test_loss}, step=best_epoch)
    wandb.finish()
    
    if args.save_model:
        torch.save(best_model, f'best_model_{args.k}.pth')
    
    # Print final metrics at best epoch
    print(f"\nBest test loss: {best_test_loss:.4f} at epoch {best_epoch}")
    print(f"\t Mean RMSE: {best_mean_rmse:.4f}")
    print(f"\t Correlation: {best_correlation:.4f}")
    # print(f"90% Coverage: {best_coverages_90:.4f} (expected: 0.90)")
    # print(f"95% Coverage: {best_coverages_95:.4f} (expected: 0.95)")

if __name__ == "__main__":
    args = get_args()
    for k in range(1):
        print(f"Running fold {k+1}/5")
        args.k = k
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