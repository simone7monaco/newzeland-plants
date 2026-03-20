import argparse
import torch
import torch.nn.functional as F
import numpy as np
from loader import PlantDataset, NormalizeFeatures, data_split
from models import TraitsPredictor, DeterministicLoss, MixedNLLLoss, graph_smoothness_loss
from train_baseline import compute_correlation
from tester import test_routine
from tqdm import trange
from pathlib import Path
import pytorch_lightning as pl
from copy import deepcopy
import wandb

import matplotlib.pyplot as plt

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
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of epochs to train the model')
    parser.add_argument('--use_env_features', type=str2bool, nargs='?', const=True, default=True, help='Whether to use environmental features')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--gnn_module', type=str, default='GATv2Conv', help="GNN attention module")#, choices=['GATConv', 'GATv2Conv', 'TransformerConv'])
    parser.add_argument('--hidden_channels', type=int, default=150, help='Number of hidden channels in the GNN')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate for the GNN')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler type', choices=[None, 'plateau', 'cosine', 'step'])

    parser.add_argument('--loss', type=str, default='dist_normal', choices=['deterministic', 'dist_normal', 'dist_lognormal'], help='Output distribution type')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Masking ratio for input features at training time')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='Weight for the KL divergence loss term')
    parser.add_argument('--smoothness_weight', type=float, default=0.0, help='Weight for the graph smoothness loss')
    parser.add_argument('--visible_loss_weight', type=float, default=0.2, help='Weight for the reconstruction loss on visible (non-masked) entries')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping (0 to disable)')
    
    parser.add_argument('--test_logging_step', type=int, default=1, help='Step for test logging')
    parser.add_argument('--save_model', action='store_true', help='Save the model after training')
    parser.add_argument('--use_wb', type=str2bool, default=False, nargs='?', const=True, help='Use Weights & Biases for logging')
    return parser.parse_args()


# --- Save per-feature error bar plots for train and test ---
def _save_error_bars(pred_mean, pred_std, true_mean, true_std, mask, out_path, title, labels=None):
    # pred_*/true_*: torch tensors shape (N, D); mask: boolean tensor shape (N, D) where True=valid
    pred_mean_np = pred_mean.detach().cpu().numpy()
    pred_std_np = pred_std.detach().cpu().numpy()
    true_mean_np = true_mean.detach().cpu().numpy()
    true_std_np = true_std.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    # For mean: compute per-feature RMSE and variability (std of absolute errors)
    mean_diff = pred_mean_np - true_mean_np
    mean_sq = mean_diff**2
    mean_rmse = np.sqrt(np.nanmean(np.where(mask_np, mean_sq, np.nan), axis=0))
    mean_var = np.nanstd(np.where(mask_np, np.abs(mean_diff), np.nan), axis=0)

    # For std: compute per-feature RMSE and variability
    std_diff = pred_std_np - true_std_np
    std_sq = std_diff**2
    std_rmse = np.sqrt(np.nanmean(np.where(mask_np, std_sq, np.nan), axis=0))
    std_var = np.nanstd(np.where(mask_np, np.abs(std_diff), np.nan), axis=0)

    D = pred_mean_np.shape[1]
    x = np.arange(D)
    fig, axes = plt.subplots(2, 1, figsize=(max(8, D * 0.25), 8))
    axes[0].bar(x, mean_rmse, yerr=mean_var, capsize=3)
    axes[0].set_title(f"{title} — Mean RMSE per feature")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("RMSE")
    if labels is not None:
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    axes[1].bar(x, std_rmse, yerr=std_var, capsize=3)
    axes[1].set_title(f"{title} — Std RMSE per feature")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("RMSE")
    if labels is not None:
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main(args):
    wandb.init(project='fern-sweep', config=args, mode='disabled' if not args.use_wb else 'online')
    
    print(f"---------------\nTraining with args: {args}")

    norm_transform = NormalizeFeatures()
    data_path = Path(f'data/Ferns/')
    dataset = PlantDataset(data_path, transform=norm_transform)
    trait_names = list(dataset.traits_mean.columns)
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
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
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

        observed_mask = ~train_data.traits_nanmask
        if model.reconstruction_mask is not None:
            masked_entries = model.reconstruction_mask  # True where observed traits were masked
            visible_entries = observed_mask & ~masked_entries  # True where observed traits are still visible
            # Full loss on masked entries (the primary denoising objective)
            loss_masked = loss_fn(pred_mean, pred_std, train_data.species_x_mean, train_data.species_x_std, masked_entries)
            # Down-weighted loss on visible entries (dense gradient signal)
            loss_visible = loss_fn(pred_mean, pred_std, train_data.species_x_mean, train_data.species_x_std, visible_entries)
            loss = loss_masked + args.visible_loss_weight * loss_visible
        else:
            loss = loss_fn(pred_mean, pred_std, train_data.species_x_mean, train_data.species_x_std, observed_mask)

        # add graph smothness loss
        gs_loss = graph_smoothness_loss(pred_mean, train_data.species_species_edge_index) * args.smoothness_weight
        loss += gs_loss
        
        if torch.isnan(loss):
            raise ValueError("Loss is NaN. Training stopped.")

        loss.backward()

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)

        log_dict = {
            'train_loss': loss.item(), 
            'graph_smoothness_loss': gs_loss.item(),
            'lr': optimizer.param_groups[0]['lr'],
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
                    'lr': optimizer.param_groups[0]['lr'],
                    # 'test_coverage_90': coverage[0.9].item(),
                    # 'test_coverage_95': coverage[0.95].item(),
                } | loss_fn.cache
                wandb.log(log_dict, step=epoch)

    # Save the model
    
    if args.save_model:
        torch.save(best_model, f'best_model_{args.k}.pth')
    
    # Print final metrics at best epoch
    print(f"\nBest test loss: {best_test_loss:.4f} at epoch {best_epoch}")
    print(f"\t Mean RMSE: {best_mean_rmse:.4f}")
    print(f"\t Correlation: {best_correlation:.4f}")

    wandb.log({'best_test_loss': best_test_loss}, step=best_epoch)
    wandb.finish()

    # ensure best model is loaded for final evaluation
    if best_model is not None:
        model.load_state_dict(best_model)

    # --- Full evaluation pipeline (Part 1 + Part 2) ---
    gen_col_names = list(dataset.traits_gen.columns)
    print("Launching full evaluation pipeline...")
    model.eval()
    test_routine(model, data, norm_transform, trait_names, device,
                 save_dir=f'results/fold_{args.k}',
                 compute_xai=True,
                 gen_col_names=gen_col_names)

    # --- Save per-feature error bar plots for train and test ---
    with torch.no_grad():
        pred_mean_train, pred_std_train = model(train_data)
        pred_mean_test, pred_std_test = model(test_data)

    train_mask = ~train_data.traits_nanmask
    test_mask = ~test_data.traits_nanmask

    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    _save_error_bars(pred_mean_train, pred_std_train, train_data.species_x_mean, train_data.species_x_std, train_mask,
                     plots_dir / f"errors_train_k{args.k}.png", f"Fold {args.k} Train", labels=trait_names)
    _save_error_bars(pred_mean_test, pred_std_test, test_data.species_x_mean, test_data.species_x_std, test_mask,
                     plots_dir / f"errors_test_k{args.k}.png", f"Fold {args.k} Test", labels=trait_names)
    

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