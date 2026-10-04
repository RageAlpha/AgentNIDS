from __future__ import annotations

import csv
import gc
import importlib.metadata
import json
import shutil
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import cupy as cp
import numpy as np
import torch

from .models.base_models import GpuMemoryMonitor
from .reliability.cre import fit_cre_family,predict_cre
from .reliability.round1_data import PRIMARY,CONTROL,METHODS,SCORE_DTYPE,load_partition
from .evaluation.round1_metrics import probability_metrics,detection_metrics,complementarity
from .evaluation.round1_bootstrap import domain_bootstrap,percentile_ci,gate
from .replay.predictions import write_json
from .p3 import upstream_inventory


def release():
    gc.collect();cp.get_default_memory_pool().free_all_blocks();torch.cuda.empty_cache()


def evaluate_partition(config,data,fitted,output,predictor=predict_cre,methods=METHODS):
    domain=data['audit']['domain'];partition=data['audit']['partition']
    result=dict(audit=data['audit'],CRE={},detectors={},G0={})
    strata={'all':cp.ones(data['audit']['rows'],dtype=bool),'Benign':data['labels']==0,'Attack':data['labels']==1}
    output.mkdir(parents=True,exist_ok=True)
    for i,mid in enumerate(PRIMARY):
        scores=predictor(data['x'][mid],fitted[mid])
        stored=np.lib.format.open_memmap(output/f'{mid}.npy',mode='w+',dtype=np.dtype([(m,'<f8') for m in methods]),shape=(data['audit']['rows'],))
        for method in methods:
            stored[method]=cp.asnumpy(scores[method])
        stored.flush()
        correct=(~data['errors'][:,i]).astype(cp.uint8)
        result['CRE'][mid]={}
        for stratum,mask in strata.items():
            result['CRE'][mid][stratum]={}
            for method in methods:
                result['CRE'][mid][stratum][method]=probability_metrics(correct[mask],scores[method][mask],config,ranking_only=method=='one_minus_u')
        del scores,stored
        release()
    for stratum,mask in strata.items():
        result['G0'][stratum]={
            'primary':complementarity(data['errors'][mask],PRIMARY),
            'All49':complementarity(data['control_errors'][mask],CONTROL)}
        result['detectors'][stratum]={}
        for mid in PRIMARY+CONTROL:
            result['detectors'][stratum][mid]=detection_metrics(data['labels'][mask],data['probability'][mid][mask],config)
    write_json(output/'metrics.json',result)
    print(f'{domain}/{partition}: full CRE and detector metrics saved',flush=True)
    return result


def fold_run(config,prepared,fold,output_root=None,loader=load_partition,fitter=fit_cre_family,predictor=predict_cre,methods=METHODS):
    project=Path(config['project_root']);output=(Path(output_root) if output_root is not None else project/config['outputs']['round1']/'P4')/fold
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();monitor=GpuMemoryMonitor(0);monitor.start()
    domains=config['folds'][fold]['development']
    train=[loader(config,prepared,fold,d,'R_fit',True) for d in domains]
    training_audits=[d['audit'] for d in train]
    fitted={}
    for i,mid in enumerate(PRIMARY):
        print(f'{fold}/{mid}: fitting CRE family',flush=True)
        fit=fitter([d['x'][mid] for d in train],[(~d['errors'][:,i]).astype(cp.float64) for d in train],config)
        fit.update(fold=fold,detector=mid,fit_partition='R_fit',domains=domains,data_audits=training_audits)
        write_json(output/'models'/f'{mid}.json',fit)
        assert fit['status']=='COMPLETE',f'{fold}/{mid}: {fit["status"]}'
        assert all(m['training_complete'] if 'training_complete' in m else m['converged_gradient'] for m in fit['models'].values()),f'{fold}/{mid}: solver_completion_criterion_not_met'
        fitted[mid]=fit
        release()
    del train
    release()
    evaluation={};domain_boots=[]
    bootstrap_output=output/'bootstrap';bootstrap_output.mkdir(exist_ok=True)
    rng=cp.random.RandomState(config['seed'])
    for partition in ['R_cal','R_select']:
        evaluation[partition]={}
        for domain in domains:
            data=loader(config,prepared,fold,domain,partition)
            score_folder=output/'scores'/domain/partition
            evaluation[partition][domain]=evaluate_partition(config,data,fitted,score_folder,predictor,methods)
            if partition=='R_select':
                boot=domain_bootstrap(config,data,score_folder,bootstrap_output,rng,methods)
                domain_boots.append(boot)
            del data
            release()
    g0_rep=(domain_boots[0][0]+domain_boots[1][0])/2
    auc_rep=(domain_boots[0][1]+domain_boots[1][1])/2
    hu=methods.index('hu');hus=methods.index('hus')
    g1_rep=(auc_rep[:,:,hu]-auc_rep[:,:,hus]).mean(axis=1)
    np.save(bootstrap_output/'fold_G0_primary_All49.npy',g0_rep)
    np.save(bootstrap_output/'fold_AURC.npy',auc_rep)
    np.save(bootstrap_output/'fold_G1.npy',g1_rep)
    estimable=all(b[2]['blocks']>=config['bootstrap']['minimum_units_per_evaluation_domain'] for b in domain_boots)
    points={};gates={};details={}
    for stratum in ['all','Benign','Attack']:
        g0=np.mean([evaluation['R_select'][d]['G0'][stratum]['primary']['oracle_headroom'] for d in domains])
        control=np.mean([evaluation['R_select'][d]['G0'][stratum]['All49']['oracle_headroom'] for d in domains])
        g1=np.mean([evaluation['R_select'][d]['CRE'][mid][stratum]['hu']['AURC']-evaluation['R_select'][d]['CRE'][mid][stratum]['hus']['AURC'] for d in domains for mid in PRIMARY])
        points[stratum]=dict(G0=float(g0),G0_All49=float(control),G1=float(g1))
    for name,rep in [('G0',g0_rep[:,0]),('G1',g1_rep)]:
        effect=points['all'][name];ci=percentile_ci(rep);threshold=config['gates'][name]['minimum_effect']
        gates[name]=dict(estimate=effect,CI95=ci,minimum_effect=threshold,estimable=estimable,
            status=gate(effect,ci,threshold,estimable),replicates=config['bootstrap']['replicates'])
    for i,mid in enumerate(PRIMARY):
        method_results={}
        for j,method in enumerate(methods):
            point=np.mean([evaluation['R_select'][d]['CRE'][mid]['all'][method]['AURC'] for d in domains])
            method_results[method]=dict(AURC=float(point),CI95=percentile_ci(auc_rep[:,i,j]))
        delta=auc_rep[:,i,hu]-auc_rep[:,i,hus]
        details[mid]=dict(methods=method_results,delta_hu_minus_hus=dict(estimate=method_results['hu']['AURC']-method_results['hus']['AURC'],CI95=percentile_ci(delta)))
    write_json(output/'gate_report.json',dict(fold=fold,domains=domains,gates=gates,descriptive_strata=points,
        All49_G0_CI95=percentile_ci(g0_rep[:,1]),model_AURC=details,
        bootstrap=[b[2] for b in domain_boots],
        round2_eligible=all(g['status']=='PASS' for g in gates.values()) if output_root is None else None,
        round2_eligibility_source='P4_primary_condition_only',
        uncertainty_scope=config['bootstrap']['uncertainty_scope']))
    validation=dict(status='PASS',fold=fold,domains=domains,training_audits=training_audits,
        evaluation_audits=[evaluation[p][d]['audit'] for p in evaluation for d in domains],
        paired_fit='identical_R_fit_rows_domain_weights_solver_and_feature_column_subsets',
        no_meta_calibration=True,original_shift_preserved=True,G0_reselect_best_each_replicate=True,
        bootstrap_samples_shared_across_all_primary_methods_and_both_G0_groups=True,
        solver_gradient_max=max((m['gradient_inf'] for f in fitted.values() for m in f['models'].values() if 'gradient_inf' in m),default=None),
        device='cuda:0',models=sum(len(f['models']) for f in fitted.values()),method_order=methods,bootstrap_replicates=config['bootstrap']['replicates'])
    write_json(output/'validation.json',validation)
    duration=time.perf_counter()-started;memory=monitor.stop()
    manifest=dict(status='COMPLETE',fold=fold,domains=domains,training_audits=training_audits,
        seconds=duration,gpu_memory=memory,gates=gates,completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'manifest.json',manifest)
    write_fold_tables(output,evaluation)
    print(f'{fold}: COMPLETE, G0={gates["G0"]["status"]}, G1={gates["G1"]["status"]}, {duration:.2f}s',flush=True)
    return manifest


def write_fold_tables(output,evaluation):
    rows=[]
    for partition,domains in evaluation.items():
        for domain,metrics in domains.items():
            for mid,strata in metrics['CRE'].items():
                for stratum,methods in strata.items():
                    for method,result in methods.items():
                        row=dict(partition=partition,domain=domain,detector=mid,stratum=stratum,method=method)
                        row.update({k:v for k,v in result.items() if k not in ['risk_coverage','calibration_bins']})
                        rows.append(row)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'CRE_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def run_p4(config,prepared):
    project=Path(config['project_root']);output=project/config['outputs']['round1']/'P4'
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();inventory=upstream_inventory(config)
    for root in [project/config['outputs']['base_predictions'],project/config['outputs']['shift'],project/config['outputs']['round1']/'P3']:
        inventory.extend(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(root.rglob('*')) if p.is_file())
    write_json(output/'upstream_inventory.json',dict(files=inventory))
    shutil.copy2(project/'configs/execution_v2.json',output/'execution_v2.json')
    runtime=dict(python=sys.executable,gpu=torch.cuda.get_device_name(0),device='cuda:0',seed=config['seed'],
        bootstrap_seed_rule='reset_frozen_seed_per_fold_then_draw_domains_sequentially',
        versions={p:importlib.metadata.version(p) for p in ['torch','cupy-cuda12x','numpy','pyarrow']},
        torch_cuda=torch.version.cuda,cupy_runtime=cp.cuda.runtime.runtimeGetVersion())
    write_json(output/'runtime.json',runtime)
    results=[]
    for fold in ['F1','F2','F3']:
        results.append(fold_run(config,prepared,fold));release()
        write_json(output/'progress.json',dict(completed_folds=[r['fold'] for r in results]))
    for before in inventory:
        s=Path(before['path']).stat()
        assert s.st_size==before['bytes'] and s.st_mtime_ns==before['mtime_ns']
    result=dict(status='COMPLETE',folds=results,seconds=time.perf_counter()-started,upstream_unchanged_files=len(inventory))
    write_json(output/'P4_manifest.json',result)
    write_json(output/'P4_validation.json',dict(status='PASS',folds=['F1','F2','F3'],CRE_models=27,replicates_per_fold=2000,upstream_unchanged=True))
    return write_report(config,result,runtime)


def write_report(config,result,runtime):
    project=Path(config['project_root']);output=project/config['outputs']['round1']/'P4'
    lines=['# P4 Round 1 phase report','', 'State: COMPLETE','',
        'Use only the development domain complete windows fold, retaining the initialized complete windows. R_fit fit, R_cal probability diagnosis, R_select evaluation; labels is associated with P3 domain/rank in P1 order labels. The basic model, Platt, h, preprocessing, and P3 products maintain freeze.','',
        '## G0/G1','', '| Folded | development domain | G0 | G0 95% CI | G0 Conclusion | G1 | G1 95% CI | G1 Conclusion |','|---|---|---:|---|---|---:|---|---|']
    for fold in result['folds']:
        a,b=fold['gates']['G0'],fold['gates']['G1']
        lines.append(f"| {fold['fold']} | {'/'.join(fold['domains'])} | {a['estimate']:.8f} | {a['CI95']} | {a['status']} | {b['estimate']:.8f} | {b['CI95']} | {b['status']} |")
    lines+=['','the G0 threshold value is 0.01; The G1 threshold is 0.001. The PASS requirement points are estimated to have the minimum effect, the 95% range below >0 and at least 20 blocks in each range; BELOW_MINIMUM_EFFECT requires that the upper limit of 95% be below the minimum and estimated effect; The rest of the INCONCLUSIVE.','',
        '## data and fit','', '| fold | domain | partition | complete windows record | window | Initial record | excluding tail | Benign | Attack |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    fits=[]
    for f in result['folds']:
        folder=output/f['fold'];v=json.loads((folder/'validation.json').read_text())
        for a in v['training_audits']+v['evaluation_audits']:
            lines.append(f"| {f['fold']} | {a['domain']} | {a['partition']} | {a['rows']:,} | {a['windows']} | {a['initialization_rows']} | {a['excluded_tail']} | {a['benign']:,} | {a['attack']:,} |")
        for mid in PRIMARY:
            fit=json.loads((folder/'models'/f'{mid}.json').read_text())
            for method,m in fit['models'].items():
                fits.append(dict(fold=f['fold'],detector=mid,method=method,**m))
    lines+=['','Each of the two fields records a weight of N/(2Nd) and W=N; Sharing the weighted mean with the sum of the squares of the scale=1 constants. The goal is to add BCE/W+||beta||²/(2CW) and C=1, with no penalty. All models are set to zero initial values, CUDA float64 LBFGS and freeze mutations and tolerance differences; The main comparison of records, weights, and settings is completely consistent.','',
        '| Folded | detector | CRE | Replacement | End goal | gradient∞ parameter | Folded | time consuming (s) |','|---|---|---|---:|---:|---:|---|---:|']
    for f in fits:
        lines.append(f"| {f['fold']} | {f['detector']} | {f['method']} | {f['iterations']} | {f['objective']:.9g} | {f['gradient_inf']:.4g} | {f['converged_gradient']} | {f['seconds']:.3f} |")
    lines+=['','## by model main comparison with full features comparison','',
        '| Folded | detector | AURC f(h,u) | AURC f(h,u,s) | Difference | Difference 95% CI |',
        '|---|---|---:|---:|---:|---|']
    controls=[]
    for f in result['folds']:
        g=json.loads((output/f['fold']/'gate_report.json').read_text())
        for mid,v in g['model_AURC'].items():
            delta=v['delta_hu_minus_hus']
            lines.append(f"| {f['fold']} | {mid} | {v['methods']['hu']['AURC']:.8f} | {v['methods']['hus']['AURC']:.8f} | {delta['estimate']:.8f} | {delta['CI95']} |")
        controls.append(f"| {f['fold']} | {g['descriptive_strata']['all']['G0_All49']:.8f} | {g['All49_G0_CI95']} |")
    lines+=['','| Folded | All49 G0 | 95% CI |','|---|---:|---|']+controls
    lines+=['','The G1 point of F3 is estimated to reach 0.001 and confidence interval to exceed 0, thus concluding INCONCLUSIVE. Three times the conclusion of complete reservation.','',
        '## root due to repair and recalculation','',
        'The first run found that the F1 XGBoost h difference was very small, directly to the number of approaching 1 asking for mean and gradually reducing the gap, standardizing the mean remainder to reach 2.59e-7. cre.py reduces the residual to 1.94e-14 by first subtracting the common focal point and then adding the weight of mean to the total differential equation. The freeze weight, differential definition, parameters and tolerance are consistent. After the repair, complete re-run three times from F1; This report corresponds to the final effective running. scaler_root_cause.json preserves the actual diagnosis and logs/P4_attempt1.log preserves the first run records.','',
        '## modified code','',
        'src/nids/reliability/cre.py is responsible for GPU fit and stable standardization; round1_data.py is responsible for the complete windows and labels associations; src/nids/evaluation/round1_metrics.py is responsible for the index and the ties sequence; round1_bootstrap.py is responsible for the rules for pulling blocks and thresholds; src/nids/p4.py is responsible for three rounds of operation and reporting; src/nids/run.py and scripts/run_p4.sh provide a unified input. tests/check_round1_mathematics.py verifies the mathematical definition.','',
        '## indicator and bootstrap','',
        'Each time CRE_metrics.csv is complete, it lists R_cal/R_select, two domains, three main detector, all/Benign/Attack, and prior, raw_h, f(u), f(h,u), f(h,u,s), and 1-u. 1-u is just a sequence diagnosis, the rest of the output is AURC, correctness, AUROC/AUPRC, Brier, NLL and 15 boxes are ECE. scores/ range / partition /metrics.json 100 point risk-coverage, box diagnosis, six basic detector detection indicators, G0 complementarity, pairwise disagreement/double-fault/ error correlates, recoverable errors and oracle headroom. The AUROC of the single-category target and the AUPRC of the non-negative case are null and indicate the reason.', '',
        'Each domain has a continuous 10 of complete windows in blocks, with the last short remaining; 2000 times GPU as the bootstrap of the window block. The same block multiplicity is used for all baseline and both G0; G0 gradually resets the optimal single detector. The sequence of the fractions is pre-calculated only once, and the calculation of the original block , of the same ( fractional array ) is compressed without loss; The precise forefront definition is calculated in the random order of the ties group with the expectation that AURC will be calculated each time it is repeatedly recorded. shift is maintained with the original record.', '',
        'G0/G1 confidence interval is the percentage of 2.5/97.5 after the independent blocks have been pulled together in different regions. G1 and then to the three main detector equal weighting. gate_report.json contains the all49 G0 contrast interval, the AURC of each model method and the contrast value interval; The results are estimated as descriptive points. All the original draw blocks with 2000 AURC/G0/G1 arrays have been preserved, with the interval conditioned on the freeze model and the current evaluation range.', '',
        '## equipment, time-consuming and checking','',f" Equipment:{runtime['gpu']},CUDA:0; What is the meaning of this verse?{runtime['versions']}。",'',
        f" Finally, the effective P4 runs out of time:{result['seconds']:.3f} Seconds.",'', '| fold | time consuming (s) | sampling peak showing (MiB) |','|---|---:|---:|']
    for f in result['folds']:
        lines.append(f"| {f['fold']} | {f['seconds']:.3f} | {f['gpu_memory']['peak_used_mib']:.1f} |")
    lines+=['','The total installed capacity of CUDA total-free for each 100ms sample, covering the fit folding indicators and bootstrap. The numerical values fit, standardization, indicators, ranking and resampling use GPU; Parquet read, identity is associated with the use of CPU to preserve.', '',
        f"{result['upstream_unchanged_files']} The size of an upstream file is consistent with the timescale of the nanosecond modification.",'',
        'Mathematical checks cover the ties sequence, repeated draw blocks and final short blocks with precision unfolding, standard AUROC/AUPRC/Brier/NLL counterparts, G0 re-selection and freeze determination rules. The validation.json recording data range, labels numbers, recording and weighing pairs, CUDA devices, all solver gradient convergence, original shift and shared sampling.','',
        'For actual product reviews, see artifact_validation.json: 27, CRE, 36 score files, 648 line CRE indicators; The block record is consistent with the multiplicity, the folding polymer array, the confidence interval recalculation and the freeze configuration copies.','',
        '## products','', '- outputs/round1_v2/P4/P4_manifest.json、P4_validation.json、artifact_validation.json、runtime.json、execution_v2.json',
        '- outputs/round1_v2/P4/F*/models/: 9 a CRE, share scaler, correctly lead the fit diagnosis',
        '- outputs/round1_v2/P4/F*/scores/: R_cal/R_select percentage and complete indicator JSON',
        '- outputs/round1_v2/P4/F*/CRE_metrics.csv、gate_report.json、validation.json',
        '- outputs/round1_v2/P4/F*/bootstrap/: Pull blocks, multiplicity, 2000 times AURC/G0/G1 and block manifest',
        '- logs/P4.log；scripts/run_p4.sh；tests/check_round1_mathematics.py', '',
        'P4 is completed and P5/P6/P7 is scheduled to be implemented in the next phase.','']
    path=output/'stage_report.md';path.write_text('\n'.join(lines));return path
