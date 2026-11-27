"""Finish missing folds of the interrupted factorial benchmark without replacing completed folds."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, os, subprocess, sys, time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'factorial_benchmark_results'
CONDITIONS={'limited_individual':('../.soft_del/FernMinMax.xlsx','individual'), 'limited_paired':('../.soft_del/FernMinMax.xlsx','paired'), 'complete_individual':('FernMinMax_all.xlsx','individual'), 'complete_paired':('FernMinMax_all.xlsx','paired')}

def fold_outputs(root,fold):
    # Historical folds wrote predictions_min_original.csv; current code writes
    # original-scale predictions_long.csv (predictions_min.csv is normalized in old runs).
    return [p for name in ('predictions_min_original.csv','predictions_long.csv') for p in root.glob(f'*/fold_{fold}/{name}')]

def run(condition,fold,workbook,mask):
    out=OUT/'recovery'/condition/f'fold_{fold}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'success.json').exists():
        print(f'Already complete {condition} fold {fold}',flush=True);return
    cmd=[sys.executable,'train.py','--traits_filename',workbook,'--split_strategy','stable','--seed','42','--occlusion_protocol',mask,'--k',str(fold),'--epochs','1000','--use_env_features','true','--use_phylo_features','true','--bounds_parameterization','independent','--compute_xai','false','--save_model','false','--output_dir',str(out)]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(OUT/'matplotlib-cache'),PYTHONUNBUFFERED='1',WANDB_MODE='disabled')
    print(f'Started {condition} fold {fold}',flush=True);start=time.monotonic()
    with (out/'run.log').open('w') as f: p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{condition} fold {fold}: exit {p.returncode}, see {out}/run.log')
    matches=fold_outputs(out,fold)
    if len(matches)!=1:raise RuntimeError(f'Missing output for {condition} fold {fold}: {matches}')
    (out/'success.json').write_text(json.dumps({'condition':condition,'fold':fold,'command':cmd,'seconds':round(time.monotonic()-start,2)},indent=2)+'\n')
    print(f'Completed {condition} fold {fold} in {(time.monotonic()-start)/60:.1f} min',flush=True)

def main():
    jobs=[]
    for condition,(workbook,mask) in CONDITIONS.items():
        original=OUT/'seed_42'/condition/'gnn'
        for fold in range(5):
            paths=fold_outputs(original,fold)
            if not paths: jobs.append((condition,fold,workbook,mask))
    print(f'Recovering {len(jobs)} missing folds; completed folds are retained',flush=True)
    errors=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in as_completed([pool.submit(run,*j) for j in jobs]):
            try:fut.result()
            except Exception as e:print('ERROR',e,flush=True);errors.append(str(e))
    if errors:raise SystemExit(f'{len(errors)} fold jobs failed')
    print('All missing folds completed',flush=True)

if __name__=='__main__':main()
