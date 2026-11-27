"""Exercise scheduling, resume and aggregation without training a benchmark."""
import argparse
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_benchmark as benchmark
from benchmark_analysis import discover_benchmark_metric_rows
from r_baselines_fit import feature_configurations, model_dir_suffix


class BenchmarkPipelineTests(unittest.TestCase):
    def test_standalone_enumerates_supported_inputs_only(self):
        args = argparse.Namespace(
            baseline_models=','.join((*benchmark.BASELINES, 'missforest')),
            all_feature_combinations=True, use_env_features=False, use_phylo_features=False,
        )
        configs = list(feature_configurations(args))
        self.assertEqual(len(configs), 4)
        for config in configs:
            names = config.baseline_models.split(',')
            if config.use_env_features or config.use_phylo_features:
                self.assertEqual(set(names), {*benchmark.COVARIATE_BASELINES, 'missforest'})
            else:
                self.assertEqual(set(names), {*benchmark.BASELINES, 'missforest'})

    def test_all_input_runs_resume_and_merge_original_units(self):
        commands = []
        scale_for_gnn = 'original'

        def fake_run(command, *, label):
            argv = shlex.split(command[-1].split('exec ', 1)[1])
            commands.append(argv)

            def value(flag):
                return argv[argv.index(flag) + 1]

            output = Path(value('--output_dir'))
            env = value('--use_env_features') == 'true'
            phylo = value('--use_phylo_features') == 'true'
            if 'r_baselines_fit.py' in argv:
                args = argparse.Namespace(
                    training_protocol=1, occlusion_protocol='paired',
                    include_unobserved_species=False, use_env_features=env,
                    use_phylo_features=phylo, use_categorical_features=True,
                )
                dirs = [(output / (model.upper() + model_dir_suffix(args, model)), False)
                        for model in value('--baseline_models').split(',')]
            else:
                dirs = [(output / 'FERN', True)]

            coverage = pd.DataFrame({'species': ['fern_a'], 'placement': ['exact binomial'], 'in_tree': [True]})
            if 'r_baselines_fit.py' in argv:
                output.mkdir(parents=True, exist_ok=True)
                coverage.to_csv(output / benchmark.TREE_COVERAGE_FILE, index=False)
            for directory, gnn in dirs:
                fold = directory / 'fold_0'
                fold.mkdir(parents=True)
                for variable, original_value in [('min', 12.0), ('max', 24.0)]:
                    # Every model, GNN included, exports original-scale predictions.
                    pd.DataFrame({'Spore': [original_value]},
                                 index=pd.Index(['fern_a'], name='species')).to_parquet(
                        directory / f'predictions_{variable}_all.parquet')
                pd.DataFrame([dict(species='fern_a', fold='fold_0', trait='Spore', variable=variable,
                                   observed=value, predicted=value, evaluated=True,
                                   scale=scale_for_gnn if gnn else 'original')
                              for variable, value in [('min', 12.0), ('max', 24.0)]]).to_parquet(
                    directory / benchmark.LONG_RESULT_FILE, index=False)
                metric_name = 'per_trait_metrics_original.csv' if gnn else 'per_trait_metrics.csv'
                pd.DataFrame([dict(trait='Spore', variable='min', n=1, RMSE=1.0,
                                   MAE=1.0, Pearson_r=0.5, Spearman_rho=0.5)]).to_csv(
                    fold / metric_name, index=False)
                if gnn:
                    coverage.to_csv(directory / benchmark.TREE_COVERAGE_FILE, index=False)
                    pd.DataFrame({'species': ['fern_a'], 'Spore_min': [12.0],
                                  'Spore_max': [24.0]}).to_csv(
                        directory / 'predictions_original_all_folds.csv', index=False)
                    for filename in benchmark.ATTRIBUTION_FILES:
                        if 'spatial' not in filename or env:
                            pd.DataFrame({'species': ['fern_a']}).to_parquet(directory / filename)

        with tempfile.TemporaryDirectory() as tmp:
            argv = ['run_benchmark.py', '1', '--output-dir', tmp]
            with patch('sys.argv', argv), patch.object(benchmark, 'run', fake_run), \
                    patch.object(benchmark, 'log_bounds_order'), \
                    patch.object(benchmark, 'r_package_versions', return_value={'R': '4.5.3'}):
                benchmark.main()
                self.assertEqual(len(commands), 8)  # four baseline jobs + four GNN jobs
                # Resume must distinguish baseline variants instead of rerunning
                # them or treating an input-mismatched result as completed.
                benchmark.main()
                self.assertEqual(len(commands), 8)

            root = Path(tmp)
            baseline_dirs = [p for p in (root / 'runs/seed_42/baselines').iterdir() if p.is_dir()]
            self.assertTrue((root / 'runs/seed_42/baselines' / benchmark.TREE_COVERAGE_FILE).is_file())
            self.assertEqual(len(baseline_dirs), 21)  # 5 references + 4 x 4 covariate models
            for env, phylo, combination in benchmark.FEATURE_COMBINATIONS:
                attribution = pd.read_parquet(root / combination / 'attributions_species_all.parquet')
                self.assertEqual(len(attribution), 1)
                for variable, expected in [('min', 12.0), ('max', 24.0)]:
                    frame = pd.read_parquet(root / combination / f'predictions_{variable}_all.parquet')
                    self.assertTrue(frame.source.eq('gnn').any())
                    self.assertEqual(len(frame), len(benchmark.BASELINES) + 1)
                    self.assertEqual(set(frame.species), {'fern_a'})
                    self.assertTrue(frame.Spore.eq(expected).all())
                    for model in benchmark.COVARIATE_BASELINES:
                        row = frame.loc[frame.model.str.lower().str.startswith(model + '_')].iloc[0]
                        self.assertEqual((bool(row.use_env_features), bool(row.use_phylo_features)), (env, phylo))

            metrics = discover_benchmark_metric_rows(root)
            rf = metrics.loc[metrics.model.str.startswith('RANDOM_FOREST_')]
            self.assertEqual(len(rf), 4)
            self.assertEqual(set(rf.configuration_id), {'baseline', 'environment', 'phylogeny', 'full'})
            self.assertEqual(len(metrics.loc[metrics.model.str.startswith('TRAINING_MEAN_')]), 4)


    def test_normalized_long_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp)
            pd.DataFrame({'scale': ['normalized'], 'evaluated': [True], 'predicted': [1.0]}).to_parquet(result / benchmark.LONG_RESULT_FILE)
            with self.assertRaisesRegex(ValueError, "expected only 'original'"):
                benchmark.check_original_scale(result)
            pd.DataFrame({'scale': ['original'] * 3, 'evaluated': [True, False, True],
                          'predicted': [1.0, float('nan'), float('inf')]}).to_parquet(result / benchmark.LONG_RESULT_FILE)
            with self.assertRaisesRegex(ValueError, '1 evaluated cells with non-finite'):
                benchmark.check_original_scale(result)
            (result / benchmark.LONG_RESULT_FILE).unlink()
            with self.assertRaises(FileNotFoundError):
                benchmark.check_original_scale(result)


if __name__ == '__main__':
    unittest.main()
