import argparse
from typing import cast
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from loader import PlantDataset, NormalizeFeatures, data_split
from models import TraitsPredictor, DeterministicLoss, MixedNLLLoss, MultiTargetDeterministicLoss, graph_smoothness_loss
from baselines_training import compute_correlation
from tester import Tester, fit_conformal_residual_bounds, fit_conformal_residual_bounds_minmax
from tqdm import trange
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from copy import deepcopy
import wandb
import optuna
import yaml

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


def create_validation_mask(observed_mask: torch.Tensor, ratio: float, seed_value: int) -> torch.Tensor:
    """Select a deterministic, per-trait validation subset while retaining training observations."""
    if not 0.0 < ratio < 1.0:
        raise ValueError('validation_mask_ratio must be between 0 and 1.')

    generator = torch.Generator(device=observed_mask.device).manual_seed(seed_value)
    validation_mask = torch.zeros_like(observed_mask, dtype=torch.bool)
    for trait_index in range(observed_mask.size(1)):
        observed_indices = torch.where(observed_mask[:, trait_index])[0]
        if observed_indices.numel() < 2:
            continue
        n_validation = min(max(1, round(observed_indices.numel() * ratio)), observed_indices.numel() - 1)
        chosen = torch.randperm(observed_indices.numel(), generator=generator, device=observed_mask.device)[:n_validation]
        validation_mask[observed_indices[chosen], trait_index] = True
    return validation_mask


def mask_trait_inputs(data: Data, mask: torch.Tensor, trait_feature_keys: list[str]) -> Data:
    """Hide selected observed trait cells from a graph while retaining their targets separately."""
    masked_data = data.clone()
    masked_data.traits_nanmask = masked_data.traits_nanmask | mask
    for trait_feature_key in trait_feature_keys:
        values = getattr(masked_data, trait_feature_key).clone()
        setattr(masked_data, trait_feature_key, values.masked_fill(mask, 0.0))
    return masked_data


def get_args():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of epochs to train the model')
    parser.add_argument('--use_env_features', type=str2bool, nargs='?', const=True, default=True, help='Whether to use environmental features')
    parser.add_argument('--use_phylo_features', type=str2bool, nargs='?', const=True, default=True, help='Whether to use phylogenetic features')
    parser.add_argument('--output_dir', type=Path, default='results/', help='Directory to save results and models')
    parser.add_argument('--trait_representation', type=str, default='min_max_range', choices=['mean_std', 'min_max_range'], help='Trait representation to use (mean/std or min/max/range)')
    parser.add_argument('--keep_range_features', type=str2bool, nargs='?', const=True, default=False, help='Whether to keep range features in the dataset (only relevant for min_max_range representation)')
    parser.add_argument('--invalid_bounds_policy', type=str, default='missing', choices=['missing', 'error', 'keep'], help='How to handle negative or inconsistent min/max/range records')

    parser.add_argument('--trait_norm_mean', type=str, default='logz', choices=['z', 'yj', 'logz'], help='Trait normalization mode for mean features')
    parser.add_argument('--trait_norm_std', type=str, default='z', choices=['z', 'yj', 'logz'], help='Trait normalization mode for std features')
    parser.add_argument('--trait_norm_min', type=str, default='yj', choices=['z', 'yj', 'logz'], help='Trait normalization mode for min features')
    parser.add_argument('--trait_norm_max', type=str, default='yj', choices=['z', 'yj', 'logz'], help='Trait normalization mode for max features')
    parser.add_argument('--trait_norm_range', type=str, default='yj', choices=['z', 'yj', 'logz'], help='Trait normalization mode for range features')
    parser.add_argument('--per_trait_loss', type=str2bool, nargs='?', const=True, default=True, help='Use per-trait loss reduction instead of flat entry-wise averaging')
    parser.add_argument('--k', type=int, default=-1, help='Fold index for cross-validation (0-4). Use -1 to perform a complete run over all folds sequentially.')
    parser.add_argument('--split_strategy', type=str, default='random', choices=['random', 'louvain'], help='Outer CV split: random is transductive; louvain requires balanced graph communities')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--gnn_module', type=str, default='GATv2Conv', help="GNN attention module")#, choices=['GATConv', 'GATv2Conv', 'TransformerConv'])
    parser.add_argument('--hidden_channels', type=int, default=150, help='Number of hidden channels in the GNN')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate for the GNN')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler type', choices=[None, 'plateau', 'cosine', 'step'])

    parser.add_argument('--loss', type=str, default='deterministic', choices=['deterministic', 'dist_normal', 'dist_lognormal'], help='Output distribution type')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Masking ratio for input features at training time')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='Weight for the KL divergence loss term')
    parser.add_argument('--smoothness_weight', type=float, default=0.0, help='Weight for the graph smoothness loss')
    parser.add_argument('--visible_loss_weight', type=float, default=0.2, help='Weight for the reconstruction loss on visible (non-masked) entries')
    parser.add_argument('--validation_mask_ratio', type=float, default=0.15, help='Fraction of observed outer-training cells held out for checkpoint selection')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping (0 to disable)')
    parser.add_argument('--compute_xai', type=str2bool, nargs='?', const=True, default=False, help='Whether to compute XAI attributions during testing (can be slow)')

    parser.add_argument('--test_logging_step', type=int, default=1, help='Step for test logging')
    parser.add_argument('--save_model', type=str2bool, nargs='?', const=True, default=True, help='Save the best fold checkpoint (default: true)')
    parser.add_argument('--use_wb', type=str2bool, default=False, nargs='?', const=True, help='Use Weights & Biases for logging')
    parser.add_argument('--run_optuna', action='store_true', help='Run an Optuna sweep from a sweep YAML file')
    parser.add_argument('--optuna_sweep', type=str, default=None, help='Path to Optuna sweep YAML file (grid format)')
    return parser.parse_args()


def main(args, tester: Tester, trial: optuna.trial.Trial | None = None) -> float:
    if args.use_wb:
        wandb.init(project='fern-sweep', config=args, mode='online')
    else:
        # keep wandb disabled when not requested to avoid clutter
        wandb.init(project='fern-sweep', config=args, mode='disabled')
    
    print(f"---------------\nTraining with args: {args}")

    norm_transform = NormalizeFeatures(
        trait_norm_mean=args.trait_norm_mean,
        trait_norm_std=args.trait_norm_std,
        trait_norm_min=args.trait_norm_min,
        trait_norm_max=args.trait_norm_max,
        trait_norm_range=args.trait_norm_range,
    )
    data_path = Path(f'data/Ferns/')
    trait_file = 'Traits.xlsx' if args.trait_representation == 'mean_std' else 'FernMinMax.xlsx'
    dataset = PlantDataset(
        data_path,
        transform=None,
        trait_representation=args.trait_representation,
        traits_filename=trait_file,
        invalid_bounds_policy=args.invalid_bounds_policy,
    )
    trait_names = dataset.trait_names
    raw_data = cast(Data, dataset[0])

    # Fit preprocessing on outer-training species only; transform the complete
    # graph afterward so evaluation uses the same fold-specific scale.
    raw_train_data, _ = data_split(raw_data, k=args.k, seed=seed, split_strategy=args.split_strategy)
    raw_train_data = cast(Data, raw_train_data)
    norm_transform.fit(raw_train_data)
    data = norm_transform(raw_data.clone())
            
    trait_feature_keys = ['species_x_mean', 'species_x_std']
    if args.trait_representation == 'min_max_range':
        trait_feature_keys = ['species_x_min', 'species_x_max', 'species_x_range']
        if not args.keep_range_features:
            trait_feature_keys.remove('species_x_range')

    model = TraitsPredictor(in_traits=len(trait_names), in_gen=data.species_x_gen.size(1), in_phylo=data.species_x_phylo.size(1),  # type: ignore
                            in_space=data.spatial_global_data.size(1), out_channels=len(trait_names),  # type: ignore
                            hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                            dropout=args.dropout, gnn_module=args.gnn_module, use_env_features=args.use_env_features,
                            use_phylo_features=args.use_phylo_features, mask_ratio=args.mask_ratio,
                            trait_feature_keys=trait_feature_keys)

    train_data, _ = data_split(data, k=args.k, seed=seed, split_strategy=args.split_strategy)


    validation_mask = create_validation_mask(~train_data.traits_nanmask, args.validation_mask_ratio, seed + args.k)
    if not validation_mask.any():
        raise ValueError('No observed cells are available for the inner validation mask.')
    validation_targets = tuple(getattr(train_data, key).clone() for key in trait_feature_keys)
    train_data = mask_trait_inputs(train_data, validation_mask, trait_feature_keys)
    validation_data = train_data.clone()

    model = model.to(device)
    train_data = train_data.to(device)
    validation_data = validation_data.to(device)
    validation_mask = validation_mask.to(device)
    validation_targets = tuple(target.to(device) for target in validation_targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    if args.trait_representation == 'min_max_range':
        if args.loss != 'deterministic':
            print('Warning: distribution losses apply only to mean/std; using deterministic min/max/range loss.\n')
        loss_fn = MultiTargetDeterministicLoss(per_trait_reduction=args.per_trait_loss)
    elif args.loss == 'deterministic':
        loss_fn = DeterministicLoss(per_trait_reduction=args.per_trait_loss)
        if args.kl_weight != 0.0:
            print("Warning: KL weight is ignored when using deterministic loss.\n")
    else:
        loss_fn = MixedNLLLoss(distribution=args.loss.split('_')[1], 
                               kl_weight=args.kl_weight,
                               per_trait_reduction=args.per_trait_loss)
    
    best_validation_loss = float('inf')
    best_model = None
    best_epoch = 0

    for epoch in trange(args.epochs, desc="Training", unit="epoch"):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(train_data)
        train_targets = tuple(getattr(train_data, key) for key in trait_feature_keys)

        observed_mask = ~train_data.traits_nanmask
        if model.reconstruction_mask is not None:
            masked_entries = model.reconstruction_mask  # True where observed traits were masked
            visible_entries = observed_mask & ~masked_entries  # True where observed traits are still visible
            # Full loss on masked entries (the primary denoising objective)
            if args.trait_representation == 'mean_std':
                loss_masked = loss_fn(*predictions, *train_targets, masked_entries)
                loss_visible = loss_fn(*predictions, *train_targets, visible_entries)
            else:
                loss_masked = loss_fn(predictions, train_targets, masked_entries)
                loss_visible = loss_fn(predictions, train_targets, visible_entries)
            # Down-weighted loss on visible entries (dense gradient signal)
            loss = loss_masked + args.visible_loss_weight * loss_visible
        else:
            loss = (
                loss_fn(*predictions, *train_targets, observed_mask)
                if args.trait_representation == 'mean_std'
                else loss_fn(predictions, train_targets, observed_mask)
            )

        # add graph smothness loss
        smoothness_prediction = predictions[0] if args.trait_representation == 'mean_std' else torch.cat(predictions, dim=1)
        gs_loss = graph_smoothness_loss(smoothness_prediction, train_data.species_species_edge_index) * args.smoothness_weight
        loss += gs_loss
        
        if torch.isnan(loss):
            raise ValueError("Loss is NaN. Training stopped.")

        loss.backward()

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step(None) if args.scheduler != 'plateau' else scheduler.step(loss)  # type: ignore

        log_dict = {
            'train_loss': loss.item(), 
            'graph_smoothness_loss': gs_loss.item(),
            'lr': optimizer.param_groups[0]['lr'],
            } | loss_fn.cache
        if args.use_wb:
            wandb.log(log_dict, step=epoch)
        
        if epoch % args.test_logging_step == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(validation_data)
                validation_loss = (
                    loss_fn(*predictions, *validation_targets, validation_mask)
                    if args.trait_representation == 'mean_std'
                    else loss_fn(predictions, validation_targets, validation_mask)
                )

                primary_prediction, primary_target = predictions[0], validation_targets[0]
                validation_rmse = F.mse_loss(
                    primary_prediction[validation_mask], primary_target[validation_mask], reduction='mean'
                ).sqrt()
                correlation_target = primary_target.clone()
                correlation_target[~validation_mask] = float('nan')
                correlation = compute_correlation(primary_prediction, correlation_target)

                if validation_loss.item() < best_validation_loss:
                    best_validation_loss = validation_loss.item()
                    best_validation_rmse = validation_rmse.item() if not torch.isnan(validation_rmse) else float('nan')
                    best_validation_correlation = float(correlation)
                    best_epoch = epoch
                    best_model = deepcopy(model.state_dict())
                
                log_dict = {
                    'validation_loss': validation_loss.item(),
                    'validation_mean_rmse': validation_rmse.item() if not torch.isnan(validation_rmse) else float('nan'),
                    'validation_correlation': float(correlation),
                    'lr': optimizer.param_groups[0]['lr'],
                } | loss_fn.cache
                if args.use_wb:
                    wandb.log(log_dict, step=epoch)

                # Report objective metric to Optuna when running under a trial
                if trial is not None:
                    try:
                        trial.report(float(validation_rmse.item()), epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    except Exception:
                        # ignore reporting errors for compatibility with non-pruning samplers
                        pass

    args.output_dir.mkdir(exist_ok=True, parents=True)        
    if args.save_model:
        torch.save(best_model, args.output_dir / f'best_model_{args.k}.pth')
    
    # Print final metrics at best epoch
    print(f"\nBest validation loss: {best_validation_loss:.4f} at epoch {best_epoch}")
    print(f"\t Mean RMSE: {best_validation_rmse:.4f}")
    print(f"\t Correlation: {best_validation_correlation:.4f}")

    if args.use_wb:
        wandb.log({'best_validation_loss': best_validation_loss}, step=best_epoch)

    # ensure best model is loaded for final evaluation
    if best_model is not None:
        model.load_state_dict(best_model)

    model.eval()
    with torch.no_grad():
        calibration_predictions = model(validation_data)
    calibration_truth_data = validation_data.clone().cpu()
    calibration_truth_data.traits_nanmask = calibration_truth_data.traits_nanmask & ~validation_mask.cpu()
    for trait_feature_key, target in zip(trait_feature_keys, validation_targets, strict=True):
        setattr(calibration_truth_data, trait_feature_key, target.cpu())

    calibration_data = validation_data.clone().cpu()
    if args.trait_representation == 'mean_std':
        calibration_data.species_x_mean = calibration_predictions[0].cpu()
        calibration_data.species_x_std = calibration_predictions[1].cpu()
        calibration_unnormalized = norm_transform.inverse(calibration_data, warn=False, soft_clip=True)
        calibration_truth = norm_transform.inverse(calibration_truth_data, warn=False).species_x_mean
        conformal_calibration = fit_conformal_residual_bounds(
            calibration_unnormalized.species_x_mean,
            calibration_unnormalized.species_x_std,
            calibration_truth,
            validation_mask.cpu(),
            trait_names,
        )
    else:
        calibration_data.species_x_min = calibration_predictions[0].cpu()
        calibration_data.species_x_max = calibration_predictions[1].cpu()
        calibration_data.species_x_range = calibration_predictions[2].cpu()
        calibration_unnormalized = norm_transform.inverse(calibration_data, warn=False, soft_clip=True)
        calibration_truth = norm_transform.inverse(calibration_truth_data, warn=False)
        conformal_calibration = fit_conformal_residual_bounds_minmax(
            calibration_unnormalized.species_x_min,
            calibration_unnormalized.species_x_max,
            calibration_unnormalized.species_x_range,
            calibration_truth.species_x_min,
            calibration_truth.species_x_max,
            calibration_truth.species_x_range,
            validation_mask.cpu(),
            trait_names,
        )

    # --- Full evaluation pipeline (Part 1 + Part 2) ---
    gen_col_names = list(dataset.traits_gen.columns)
    environment_feature_names = getattr(data, 'spatial_global_feature_names', None)
    env_col_names = None
    if environment_feature_names is not None and len(environment_feature_names) == data.spatial_global_data.size(1):
        env_col_names = [f"pos_{index}" for index in range(data.spatial_x.size(1))] + list(environment_feature_names)
    print("Launching full evaluation pipeline...")
    model.eval()
    tester.test_routine(model, data, norm_transform, trait_names, device,
                        save_dir=args.output_dir / f'fold_{args.k}',
                        compute_xai=args.compute_xai,
                        gen_col_names=gen_col_names,
                        env_col_names=env_col_names,
                        conformal_calibration=conformal_calibration,
                       )
    if args.use_wb:
        wandb.finish()
    # return main metric for Optuna
    return float(best_validation_rmse) if 'best_validation_rmse' in locals() else float('nan')
    

if __name__ == "__main__":
    args = get_args()
    exp_name = "fern_"
    if args.use_env_features:
        exp_name += "env_"
    if args.use_phylo_features:
        exp_name += "phylo_"
    if not args.use_env_features and not args.use_phylo_features:
        exp_name += "base"
    if args.trait_representation == 'mean_std':
        exp_name += f"_mean{args.trait_norm_mean}_std{args.trait_norm_std}"
    else:
        exp_name += (
            f"_min{args.trait_norm_min}_max{args.trait_norm_max}"
            f"_range{args.trait_norm_range}"
        )
    exp_name += f"_split{args.split_strategy}"
    if args.trait_representation == 'min_max_range':
        exp_name += f"_bounds{args.invalid_bounds_policy}"
    exp_name += "_ptr" if args.per_trait_loss else "_flat"
    if exp_name.endswith('_'):
        exp_name = exp_name[:-1]
    args.output_dir = Path(args.output_dir) / exp_name.upper()

    print(f"""
+------------------------------------------------------
| Starting training with config:
|   Epochs: {args.epochs}
|   Learning Rate: {args.lr}
|   ...
|   Use Env Features: {args.use_env_features}
|   Use Phylo Features: {args.use_phylo_features}
|   
""")
    tester = Tester(device)

    if args.run_optuna and args.optuna_sweep is not None:
        # Load sweep YAML and run Optuna GridSampler
        with open(args.optuna_sweep, 'r') as f:
            sweep_cfg = yaml.safe_load(f)

        method = sweep_cfg.get('method', 'grid')
        metric = sweep_cfg.get('metric', {})
        direction = 'minimize' if metric.get('goal', 'minimize') == 'minimize' else 'maximize'
        params = sweep_cfg.get('parameters', {})

        # Build search space dict for GridSampler: name -> list(values)
        search_space = {}
        for name, spec in params.items():
            if 'values' in spec:
                search_space[name] = spec['values']
            elif 'value' in spec:
                search_space[name] = [spec['value']]
            else:
                # unsupported spec; skip
                continue

        if method != 'grid':
            raise NotImplementedError('Only grid method is implemented for Optuna sweeps')

        sampler = optuna.samplers.GridSampler(search_space)
        study = optuna.create_study(direction=direction, sampler=sampler)

        def objective(trial: optuna.trial.Trial):
            # Build args copy and override with trial params
            local_args = deepcopy(args)
            # Ensure optuna flags off for nested runs
            local_args.run_optuna = False
            local_args.optuna_sweep = None

            for name in search_space:
                val = trial.suggest_categorical(name, search_space[name])
                # convert common types
                if name in ['per_trait_loss']:
                    if isinstance(val, str):
                        v = True if val.lower() in ('true', '1', 'yes') else False
                    else:
                        v = bool(val)
                    setattr(local_args, name, v)
                elif name in ['k']:
                    setattr(local_args, name, int(val))
                else:
                    setattr(local_args, name, val)

            local_tester = Tester(device)
            metric_val = main(local_args, tester=local_tester, trial=trial)
            return float(metric_val)

        study.optimize(objective)
        print('Best trial: ', study.best_trial.params)
    else:
        if args.k != -1:
            print(f"Running fold {args.k+1}/5")
            main(args, tester=tester)
        else:
            for k in range(5):
                print(f"Running fold {k+1}/5")
                args.k = k
                main(args, tester=tester)

        # Finally, concatenate all csv files in results/fold_*/ {attributions_spatial, attributions_species, predictions_mean, predictions_std} into single csv files in results/ for easier analysis
        tester.save_merged_original_predictions(args.output_dir)
        prediction_file_types = ['predictions_mean', 'predictions_std'] if args.trait_representation == 'mean_std' else ['predictions_min', 'predictions_max', 'predictions_range']
        if args.compute_xai:
            prediction_file_types += ['attributions_species']
            if args.use_env_features:
                prediction_file_types += ['attributions_spatial']

        for file_type in prediction_file_types:
            df_cat = pd.concat([pd.read_csv(subdir / f"{file_type}.csv") for subdir in args.output_dir.glob('fold_*')], ignore_index=True)
            df_cat.to_csv(args.output_dir / f"{file_type}_all.csv", index=False)
            print(f"Saved concatenated {file_type} to {args.output_dir / f'{file_type}_all.csv'}")
        tester.sanity_checks()