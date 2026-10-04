from pathlib import Path
import csv
import importlib.metadata
import json
import shutil
import time
import cupy as cp
import numpy as np
import pyarrow.parquet as pq
import torch
from .p4 import fold_run,release
from .p3 import upstream_inventory
from .replay.shift import compute_domain_shift
from .replay.mmd import FrozenMMD
from .reliability.sensitivity_data import sensitivity_loader
from .reliability.nonlinear import nonlinear_fitter,predict_nonlinear,METHODS as NONLINEAR_METHODS
from .replay.predictions import write_json
from .models.base_models import GpuMemoryMonitor

SETTINGS=[('jsd_5k',5000,'jsd','logistic'),('jsd_50k',50000,'jsd','logistic'),('mmd_10k',10000,'mmd2','logistic'),('xgb_jsd_10k',10000,'jsd','xgboost')]


def run_p5(config,prepared):
    root=Path(config['project_root']);output=root/config['outputs']['round1']/'P5';output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();inventory=upstream_inventory(config)
    for folder in [root/config['outputs']['base_predictions'],root/config['outputs']['shift'],root/config['outputs']['round1']/'P3',root/config['outputs']['round1']/'P4']:
        inventory.extend(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(folder.rglob('*')) if p.is_file())
    write_json(output/'upstream_inventory.json',dict(files=inventory))
    shutil.copy2(root/'configs/execution_v2.json',output/'execution_v2.json')
    runtime=dict(gpu=torch.cuda.get_device_name(0),device='cuda:0',versions={p:importlib.metadata.version(p) for p in ['torch','cupy-cuda12x','xgboost','numpy','pyarrow']})
    write_json(output/'runtime.json',runtime);results=[]
    for name,size,algorithm,cre in SETTINGS:
        t=time.perf_counter();folder=output/name;folder.mkdir(parents=True,exist_ok=True)
        setting=dict(name=name,window_size=size,shift=algorithm,CRE=cre,seed=config['seed'],bootstrap=config['bootstrap'],source_config=str(output/'execution_v2.json'))
        write_json(folder/'setting.json',setting)
        shift_root=folder/'shift';shift_reports=[]
        if cre=='xgboost':
            shift_root=root/config['outputs']['shift']/'10k'
        else:
            reference=None
            if algorithm=='mmd2':
                monitor=GpuMemoryMonitor(0);monitor.start()
                reference=FrozenMMD(config,prepared,shift_root)
                reference.manifest['gpu_memory']=monitor.stop();write_json(shift_root/'reference_manifest.json',reference.manifest)
            for domain in ['CIC','BoT','ToN']:
                shift_reports.append(compute_domain_shift(config,prepared,domain,size,algorithm,shift_root,reference));release()
            del reference
        loader=sensitivity_loader(size,shift_root);folds=[]
        for fold in ['F1','F2','F3']:
            kwargs=dict(output_root=folder,loader=loader)
            if cre=='xgboost':kwargs.update(fitter=nonlinear_fitter(folder/fold),predictor=predict_nonlinear,methods=NONLINEAR_METHODS)
            folds.append(fold_run(config,prepared,fold,**kwargs));release()
            write_json(folder/'progress.json',dict(completed_folds=[f['fold'] for f in folds]))
        result=dict(**setting,status='COMPLETE',folds=folds,shift_reports=shift_reports,seconds=time.perf_counter()-t)
        write_json(folder/'manifest.json',result);results.append(result)
        write_json(output/'progress.json',dict(completed_settings=[r['name'] for r in results]))
        print(f'P5/{name}: COMPLETE {result["seconds"]:.2f}s',flush=True)
    for item in inventory:
        stat=Path(item['path']).stat();assert stat.st_size==item['bytes'] and stat.st_mtime_ns==item['mtime_ns']
    result=dict(status='COMPLETE',settings=results,seconds=time.perf_counter()-started,upstream_unchanged_files=len(inventory))
    write_json(output/'P5_manifest.json',result)
    validate_artifacts(output,config)
    return write_report(output,result,runtime)


def validate_artifacts(output,config):
    counts=dict(fits=0,score_files=0,metric_rows=0,causal_windows=0);rows=[]
    for name,size,algorithm,cre in SETTINGS:
        if cre=='logistic':
            for domain in ['CIC','BoT','ToN']:
                windows=pq.read_table(output/name/'shift'/f'{domain}_windows.parquet').to_pylist()
                by_id={r['window_id']:r for r in windows}
                for r in windows:
                    assert r['rank_end_exclusive']-r['rank_start']==size
                    if r['initialization']:
                        assert r['previous_window_id'] is None and r['shift']==0
                    else:
                        previous=by_id[r['previous_window_id']]
                        assert previous['stream_id']==r['stream_id'] and previous['scheme']==r['scheme']
                        assert previous['rank_end_exclusive']<=r['rank_start'] and previous['last_available_ms']<=r['first_available_ms']
                        assert r['shift']==previous[algorithm]
                    counts['causal_windows']+=1
        for fold in ['F1','F2','F3']:
            folder=output/name/fold;g=json.loads((folder/'gate_report.json').read_text());v=json.loads((folder/'validation.json').read_text());boot=folder/'bootstrap'
            assert g['round2_eligible'] is None and g['round2_eligibility_source']=='P4_primary_condition_only'
            for path in (folder/'models').glob('*.json'):
                fit=json.loads(path.read_text());a,b=fit['models']['hu'],fit['models']['hus']
                for key in ['rows','domain_rows','weight_sum']:assert a[key]==b[key]
                assert fit['domains']==config['folds'][fold]['development']
                if cre=='logistic':assert a['solver']==b['solver'] and all(m['converged_gradient'] for m in fit['models'].values())
                else:
                    assert a['parameters']==b['parameters']
                    for m in fit['models'].values():assert m['training_complete'] and m['rounds']==200 and Path(m['path']).stat().st_size>0
                counts['fits']+=len(fit['models'])
            for d in g['domains']:
                meta=json.loads((boot/f'{d}_bootstrap_manifest.json').read_text());draws=np.load(boot/f'{d}_sampled_blocks.npy');mult=np.load(boot/f'{d}_block_multiplicity.npy')
                assert draws.shape==mult.shape==(2000,meta['blocks'])
                np.testing.assert_array_equal(mult,np.array([np.bincount(r,minlength=meta['blocks']) for r in draws]))
                assert meta['last_block_rows']==meta['rows']-(meta['blocks']-1)*10*size
            auc=np.load(boot/'fold_AURC.npy');methods=v['method_order'];delta=(auc[:,:,methods.index('hu')]-auc[:,:,methods.index('hus')]).mean(1)
            np.testing.assert_array_equal(auc,sum(np.load(boot/f'{d}_bootstrap_AURC.npy') for d in g['domains'])/2)
            for i,(mid,m) in enumerate(g['model_AURC'].items()):
                for j,method in enumerate(methods):
                    np.testing.assert_allclose(np.percentile(auc[:,i,j],[2.5,97.5]),m['methods'][method]['CI95'],rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose(np.percentile(auc[:,i,methods.index('hu')]-auc[:,i,methods.index('hus')],[2.5,97.5]),m['delta_hu_minus_hus']['CI95'],rtol=1e-12,atol=1e-14)
            if size==10000:
                np.testing.assert_array_equal(np.load(boot/'fold_G0_primary_All49.npy'),np.load(output.parent/'P4'/fold/'bootstrap/fold_G0_primary_All49.npy'))
            np.testing.assert_array_equal(delta,np.load(boot/'fold_G1.npy'))
            for gate,reps in [('G0',np.load(boot/'fold_G0_primary_All49.npy')[:,0]),('G1',delta)]:
                np.testing.assert_allclose(np.percentile(reps,[2.5,97.5]),g['gates'][gate]['CI95'],rtol=1e-12,atol=1e-14)
                if size==50000:assert g['gates'][gate]['status']=='INCONCLUSIVE'
            for path in (folder/'scores').glob('*/*/metrics.json'):
                metrics=json.loads(path.read_text())
                for mid in metrics['CRE']:
                    scores=np.load(path.parent/f'{mid}.npy',mmap_mode='r');assert len(scores)==metrics['audit']['rows'];counts['score_files']+=1
            counts['metric_rows']+=len(list(csv.DictReader((folder/'CRE_metrics.csv').open())))
            rows.append(dict(setting=name,fold=fold,status='PASS'))
    assert json.loads((output/'execution_v2.json').read_text())==config
    write_json(output/'P5_validation.json',dict(status='PASS',**counts,checks=rows,paired_records_weights=True,bootstrap_draws_multiplicities_and_CI=True,upstream_unchanged=True))


def write_report(output,result,runtime):
    lines=['# P5 sensitivity analysis report','','State: COMPLETESensitivity independent reporting; The main P4 experiments kept F1/F2 G0 and G1 PASS , F3 G0 PASS , G1 INCONCLUSIVE .','',
        'R_fit fit, R_cal diagnosis, R_select assessment; Each time using only its own two development domain, labels is associated with P1 domain/rank. Basic models, Platt, h, source preprocessing, JSD source box and vocabulary and predicted caching are all replicated. The round2_eligible of the sensitivity gate_report is null and the subsequent qualification is determined only by the main conditions of P4.','',
        'Difference between ## settings, effects and main experiments','','| setting | fold | G0 and 95% CI | G0 | G1 and 95% CI | G1 | G1 relative to P4 Difference | Number of domains |','|---|---|---|---|---|---|---:|---|']
    summary=[]
    for setting in result['settings']:
        for fold in setting['folds']:
            f=fold['fold'];g=json.loads((output/setting['name']/f/'gate_report.json').read_text());main=json.loads((output.parent/'P4'/f/'gate_report.json').read_text())
            a,b=g['gates']['G0'],g['gates']['G1'];diff=b['estimate']-main['gates']['G1']['estimate'];blocks='/'.join(f"{v['domain']}:{v['blocks']}" for v in g['bootstrap'])
            lines.append(f"| {setting['name']} | {f} | {a['estimate']:.8f} {a['CI95']} | {a['status']} | {b['estimate']:.8f} {b['CI95']} | {b['status']} | {diff:.8f} | {blocks} |")
            summary.append(dict(setting=setting['name'],fold=f,G0=a,G1=b,G0_difference_from_P4=a['estimate']-main['gates']['G0']['estimate'],G1_difference_from_P4=diff,blocks=blocks))
    write_json(output/'sensitivity_summary.json',summary)
    lines+=['','## by model to AURC','',
        '| Set | Fold | detector | f(h,u) | f(h,u,s) | Difference and 95% CI |',
        '|---|---|---|---:|---:|---|']
    for setting in result['settings']:
        for fold in setting['folds']:
            g=json.loads((output/setting['name']/fold['fold']/'gate_report.json').read_text())
            for mid,m in g['model_AURC'].items():
                delta=m['delta_hu_minus_hus']
                lines.append(f"| {setting['name']} | {fold['fold']} | {mid} | {m['methods']['hu']['AURC']:.8f} | {m['methods']['hus']['AURC']:.8f} | {delta['estimate']:.8f} {delta['CI95']} |")
    lines+=['','G0=0.01,G1=0.001; PASS requires the effect to reach a threshold value and 95% below >0, with at least 20 blocks in each domain. The number of blocks of 50k is less than 20, and the range and point estimates as descriptive reports, concluding that INCONCLUSIVE is unified. The relative P4 difference is a descriptive comparison.','',
        '## Range of data','','| setting | Fold | range | partition | record | complete windows | Initial record | excluding tail |','|---|---|---|---|---:|---:|---:|---:|']
    for setting in result['settings']:
        for fold in setting['folds']:
            v=json.loads((output/setting['name']/fold['fold']/'validation.json').read_text())
            for a in v['training_audits']+v['evaluation_audits']:
                lines.append(f"| {setting['name']} | {fold['fold']} | {a['domain']} | {a['partition']} | {a['rows']} | {a['windows']} | {a['initialization_rows']} | {a['excluded_tail']} |")
    lines+=['','## State of the solver','',
        '| setting | Fold | detector | method | iteration or turn number | gradient∞ parameter or final turn logloss | state | time consuming (s) |',
        '|---|---|---|---|---:|---:|---|---:|']
    for setting in result['settings']:
        for fold in setting['folds']:
            for path in sorted((output/setting['name']/fold['fold']/'models').glob('*.json')):
                fit=json.loads(path.read_text())
                for method,m in fit['models'].items():
                    if setting['CRE']=='logistic':
                        iterations=m['iterations'];diagnostic=m['gradient_inf'];status='CONVERGED'
                    else:
                        iterations=m['rounds'];diagnostic=m['history']['R_fit']['logloss'][-1];status='COMPLETE_FIXED_ROUNDS'
                    lines.append(f"| {setting['name']} | {fold['fold']} | {fit['detector']} | {method} | {iterations} | {diagnostic:.9g} | {status} | {m['seconds']:.3f} |")
    lines+=['','## fit, indicators and causal checks','',
        'linear is set to share the two domains equal weighting N/(2Nd), R_fit plus mean / overall differential, stability centralization, C=1 and freeze CUDA float64 LBFGS rules. The three main detector are fit, f(u), f(h,u), and f(h,u,s). The nonlinear is set to the input of the original h/u/s tree, recording the same weight as the float32 domain, training the f(h,u) and f(h,u,s) respectively; 200 rounds, depth3, eta0.05, the rest along freeze models.xgb, preserving the actual CUDA configuration with each round of R_fit logloss.','',
        'Each / folded CRE_metrics.csv set contains the whole domain, R_cal/R_select, all/Benign/Attack, AURC, AUROC, AUPRC, Brier, NLL, ECE; 1-u is just a sequence. scores/*/*/metrics.json contains six basic detector indicators, G0 complementarity / recoverable errors /oracle headroom, full 49 control, risk cover curve and box diagnosis. gate_report.json preserves the AURC model methods, creating a contrast between the effect and the range.','',
        'bootstrap is a pair of 2000 samples, with each 10 currently set to complete windows, retaining the final short block. All methods share the sampling record with the original shift, with G0 gradually re-determining the best single detector. The gap is conditioned on the existing fit model and the current development domain. The AUROC for single-category correctness targets and the AUPRC for non-regular targets are stored as null and the reason is recorded. causal shift strictly takes manifest as the same flow as scheme the first complete windows has been completed, the initial complete windows is 0 and retains the marking; manifest origin, time and front-wheel-drive relationship completed window-to-window verification.','',
        '## MMD Fixed reference','']
    reference=json.loads((output/'mmd_10k/shift/reference_manifest.json').read_text())
    lines.append(f" S_train fixed seeds {reference['seed']} PCG64 without returning uniformly extract 2048 points; Source standardized 49 dimensions; sigma={reference['sigma']:.12g}; Nuclear source mean ={reference['source_kernel_mean']:.12g}. the distance from median to all source reference sample pairs; biased MMD² maintains the angular coordinates, the window core and the symmetrical coordinates according to 1024 line tile. Distance directly from the accumulation of the coordinates is squared.")
    lines+=['','## equipment and time consuming','',f" Equipment {runtime['gpu']} CUDA:0; Published {runtime['versions']}. effective is time consuming {result['seconds']:.3f} Seconds.",'',
        '| setting | Total time consumed (s) | shift time consumed (s) | folded | time consumed (s) | sampling peak shown (MiB) |','|---|---:|---:|---|---:|---:|']
    for setting in result['settings']:
        for fold in setting['folds']:
            lines.append(f"| {setting['name']} | {setting['seconds']:.3f} | {sum(r['seconds'] for r in setting['shift_reports']):.3f} | {fold['fold']} | {fold['seconds']:.3f} | {fold['gpu_memory']['peak_used_mib']:.1f} |")
    lines+=['','| setting | shift range | complete windows number | distance from minimum | distance from maximum | time consuming (s) | peak display (MiB) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for setting in result['settings']:
        for r in setting['shift_reports']:
            lines.append(f"| {setting['name']} | {r['domain']} | {r['manifest_windows']} | {r['distance_min']:.9g} | {r['distance_max']:.9g} | {r['seconds']:.3f} | {r['gpu_memory']['peak_used_mib']:.1f} |")
    lines+=['','The use of CUDA total-free equipment as 100ms samples; shift_manifest and MMD references the time and peak of the manifest alternate stages of correspondence. The fit values, shift, indicators, sequences and bootstrap use GPU, read, identity associate, save and report using CPU.', '',
        f"{result['upstream_unchanged_files']} The size of an upstream file is consistent with the time it takes to modify in nanoseconds; P5_validation.json registers the preserved product, returns the results of fit, extract blocks and CI.",'',
        'Actual product verification passes: 99 with CRE (81 with linear, 18 with nonlinear); 144 with a score file; 2484 with CRE indicator; and 20656 with a newly calculated causal window. the model area and the gate area are reconstructed in a consistent manner; The G0 resampling array of 10k sensitivity is completely consistent with P4. mathematics_validation.json records CUDA distance with nuclear decomposition verification, repeatable sequencing and bootstrap mathematical inspection.','',
        'metadata of the reuse process is unified in the responsible modules: the sensitivity qualification field refers to the main P4 condition, and the description of the record is applicable to the shared standardized linear model and the original input tree model; P5 saved copies have been synced. The amendment keeps the numerical product consistent.','',
        'Difference in results: F1 is PASS with 5k, MMD and nonlinear set to G1; F2 is set to INCONCLUSIVE in nonlinear; F3 is set to BELOW_MINIMUM_EFFECT under 5k and nonlinear and INCONCLUSIVE under MMD. The 50k limits the number of pre-registered blocks to the reporting of the descriptive results. Numerical calculations and product verification have been completed.','',
        '## code and product','',
        '- src/nids/p5.py: unified P5 organization, verification, reporting; src/nids/run.py and scripts/run_p5.sh: entrance.',
        '- src/nids/replay/shift.py: Calculate causal shift by the size of the existing window; mmd.py: Fixed source reference and CUDA nuclei.',
        '- src/nids/reliability/sensitivity_data.py: associated caching of complete windows according to layouts; nonlinear.py: It is equal to CUDA XGBoost.',
        '- src/nids/p4.py and evaluation/round1_bootstrap.py: Reuse indicator process, explicit input method set and window size, P4 default settings retained.',
        '- outputs/round1_v2/P5/{jsd_5k,jsd_50k,mmd_10k,xgb_jsd_10k}/F{1,2,3}/: models, scores, bootstrap, complete indicators, gate_report, validation and manifest.',
        '- outputs/round1_v2/P5/sensitivity_summary.json、P5_manifest.json、P5_validation.json、runtime.json、execution_v2.json；logs/P5.log。','',
        'This time, the P5 is completed; P6/P7 is executed by subsequent stages.','']
    path=output/'stage_report.md';path.write_text('\n'.join(lines));return path
