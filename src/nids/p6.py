import itertools
import json
import time
import shutil
from pathlib import Path
import cupy as cp
import numpy as np
import torch
from .p3 import upstream_inventory
from .p4 import release
from .coordination.evidence import Development
from .coordination.latency import benchmark,benchmark_coordinator,benchmark_method_overheads
from .coordination.fitting import train_models
from .coordination.replay import trajectory,TRACE
from .coordination.selection import draws_for_domains,select_thresholds
from .coordination.validation import check_visibility,validate_saved
from .replay.predictions import write_json
from .models.base_models import GpuMemoryMonitor
from .evaluation.round1_metrics import probability_metrics


def initial_detector(c,fold):
    root=Path(c['project_root'])/c['outputs']['round1']/'P4'/fold
    from .reliability.round1_data import PRIMARY
    errors=[]
    for mid in PRIMARY:
        rates=[]
        for domain in c['folds'][fold]['development']:
            m=json.loads((root/'scores'/domain/'R_cal/metrics.json').read_text());rates.append(1-m['detectors']['all'][mid]['accuracy'])
        errors.append(sum(rates)/2)
    order=sorted(range(3),key=lambda i:(errors[i],i));return dict(selected=order[0],order=order,equal_domain_errors=errors,partition='R_cal')


def budget_match(records,reference_cost):
    output=[]
    for name,points in records.items():
        for budget in [1+i*.25 for i in range(9)]:
            feasible=[]
            for a,b in itertools.combinations_with_replacement(points,2):
                for k in range(101):
                    w=k/100;v={key:w*a['summary'][key]+(1-w)*b['summary'][key] for key in ['coverage','accepted_error_mass','mean_calls','normalized_cost']};v['risk']=v['accepted_error_mass']/v['coverage'] if v['coverage'] else 1
                    if v['coverage']>=.1 and v['risk']<=.05 and v['mean_calls']<=budget+1e-12 and v['normalized_cost']<=budget/3*reference_cost+1e-12:feasible.append(dict(**v,cap_a=a['cap'],cap_b=b['cap'],lambda_a=w))
            best=min(feasible,key=lambda v:(-v['coverage'],v['risk'],v['normalized_cost'],v['cap_a'],v['cap_b'],v['lambda_a'])) if feasible else None
            output.append(dict(method=name,calls_budget=budget,cost_budget=budget/3*reference_cost,status='FROZEN_EXPECTED_RANDOMIZED_POLICY' if best else 'NO_FEASIBLE_MATCHED_OPERATING_POINT',selection=best))
    return output


def freeze_trace(path,decisions,output):
    source=np.load(path,mmap_mode='r');d=np.load(decisions);target=np.lib.format.open_memmap(output,mode='w+',dtype=TRACE,shape=source.shape)
    for a in range(0,len(source),262144):
        b=min(a+262144,len(source));v=source[a:b].copy();stage=d['stage'][a:b]
        for step in range(3):
            absent=stage<step
            for key in TRACE.names:
                if key=='utility':
                    if step<2:v[key][stage<=step,step]=0
                else:v[key][absent,step]=-1 if key=='query' else 0
        target[a:b]=v
    target.flush()


def run_p6(c,prepared,reuse_fitted_folds=()):
    root=Path(c['project_root']);output=root/c['outputs']['round2']/'P6';output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    p6=json.loads((root/'configs/round2.json').read_text());shutil.copy2(root/'configs/round2.json',output/'round2_rules.json');shutil.copy2(root/'configs/execution_v2.json',output/'execution_v2.json')
    inventory=upstream_inventory(c)
    for folder in [root/c['outputs']['base_predictions'],root/c['outputs']['shift'],root/c['outputs']['round1']/'P4',root/c['outputs']['round1']/'P5']:
        inventory.extend(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in folder.rglob('*') if p.is_file())
    write_json(output/'upstream_inventory.json',dict(files=inventory));write_json(output/'F3_branch.json',dict(status='NOT_ELIGIBLE',G0='PASS',G1='INCONCLUSIVE',development_executed=False))
    results=[]
    for fold in ['F1','F2']:
        folder=output/fold;folder.mkdir(exist_ok=True);t=time.perf_counter();monitor=GpuMemoryMonitor(0);monitor.start()
        data={part:[Development(c,prepared,fold,d,part) for d in c['folds'][fold]['development']] for part in ['R_fit','R_cal','R_select']}
        initial=initial_detector(c,fold);write_json(folder/'initial_detector.json',initial);print(f'{fold}: initial={initial}',flush=True)
        latency=benchmark(c,data['R_cal'],folder)
        if fold in reuse_fitted_folds:
            fitted=json.loads((folder/'models/fitted.json').read_text())
            reused_fit_seconds=sum(json.loads(path.read_text())['seconds'] for path in (folder/'models').glob('*.json') if path.name!='fitted.json')
        else:
            fitted=train_models(data['R_fit'],c,folder);reused_fit_seconds=0.
        release()
        write_json(folder/'fit_audits.json',dict(domains=[d.audit for d in data['R_fit']]))
        write_json(folder/'visibility_validation.json',check_visibility(fitted,latency['normalized_cost']));benchmark_coordinator(fitted,data['R_cal'],folder)
        latency=benchmark_method_overheads(fitted,data['R_cal'],initial,latency,folder)
        latency['shift_normalized']={}
        for d in data['R_cal']:
            shift=json.loads((root/c['outputs']['shift']/'10k'/f'{d.domain}_manifest.json').read_text())
            latency['shift_normalized'][d.domain]=shift['seconds']*1000/(shift['unique_data_windows']*10000)/latency['normalization_ms']
        latency['budget_reference_full_cost']+=max(latency['shift_normalized'].values());write_json(folder/'cost_model.json',latency)
        methods=[('coordinator','coordinator',None),('learned_cascade','learned',None),('random','random',None),('cheapest','cheapest',None),('uncertainty','uncertainty',None),('best_single','best',None),('uniform','uniform',None),('static','static',None),('confidence','confidence',None),('soft_moe','soft',None),('hard_moe','hard',None)]+[(f'fixed_{"".join(map(str,order))}','fixed',order) for order in itertools.permutations(range(3))]
        draws=draws_for_domains(data['R_select'],c['seed'])
        for d,(mult,sizes) in zip(data['R_select'],draws):np.savez(folder/f'{d.domain}_paired_blocks.npz',multiplicity=cp.asnumpy(mult),block_rows=cp.asnumpy(sizes))
        policies={};diagnostics={};trace_reports=[]
        for name,kind,order in methods:
            policy=folder/'policies'/name;policy.mkdir(parents=True,exist_ok=True);paths=[]
            for partition in ['R_cal','R_select']:
                for d in data[partition]:
                    path=policy/f'{d.domain}_{partition}_trajectory.npy';trace_reports.append(trajectory(d,fitted,latency,initial,kind,path,order))
                    if partition=='R_select':paths.append(path)
                    else:
                        tr=np.load(path,mmap_mode='r');prob=cp.asarray(tr['fused'][:,-1]);r=cp.asarray(tr['reliability'][:,-1]);correct=((prob>=.5)==d.labels()).astype(cp.uint8)
                        diagnostics[f'{name}/{d.domain}']=probability_metrics(correct,r,c,ranking_only=kind in ['uniform','static','confidence','soft'])
                    release()
            caps=[1] if kind=='best' else [1,2] if kind=='hard' else [1,2,3];policies[name]=[]
            for cap in caps:
                selection=select_thresholds(data['R_select'],paths,cap,draws,policy,allow_stop=kind in ['coordinator','fixed','learned','random','cheapest','uncertainty']);policies[name].append(selection)
                for d,path in zip(data['R_select'],paths):freeze_trace(path,policy/f'{d.domain}_cap{cap}_decisions.npz',policy/f'{d.domain}_cap{cap}_frozen_trace.npy')
            print(f'{fold}/{name}: frozen {[ (p["cap"],p["tau_stop"],p["tau_final"]) for p in policies[name]]}',flush=True)
        fixed={str(cap):min([(name,p) for name,points in policies.items() if name.startswith('fixed_') for p in points if p['cap']==cap],key=lambda v:(-v[1]['summary']['coverage'],v[1]['summary']['risk'] if v[1]['summary']['risk'] is not None else 1,v[1]['summary']['normalized_cost'],v[0]))[0] for cap in [1,2,3]}
        write_json(folder/'fixed_cascade_selection.json',fixed);write_json(folder/'R_cal_diagnostics.json',diagnostics);write_json(folder/'budget_matched.json',budget_match(policies,latency['budget_reference_full_cost']))
        frozen={name:[{k:v for k,v in p.items() if k not in ['stop_candidates','final_candidates']} for p in points] for name,points in policies.items()};write_json(folder/'frozen_policies.json',frozen)
        result=dict(fold=fold,status='FROZEN',reused_fit_seconds=reused_fit_seconds,seconds=time.perf_counter()-t+reused_fit_seconds,gpu_memory=monitor.stop(),initial=initial,methods=len(methods),policies=sum(map(len,policies.values())),audits={part:[d.audit for d in ds] for part,ds in data.items()},trace_reports=trace_reports)
        write_json(folder/'manifest.json',result);results.append(result);release()
    for item in inventory:
        stat=Path(item['path']).stat();assert stat.st_size==item['bytes'] and stat.st_mtime_ns==item['mtime_ns']
    validation=validate_saved(output);write_json(output/'development_validation.json',validation)
    result=dict(status='DEVELOPMENT_SELECTION_COMPLETE_PENDING_CONDITIONAL_REPLAY',folds=results,F3='NOT_ELIGIBLE',outer_results_opened=False,seconds=time.perf_counter()-start+sum(f['reused_fit_seconds'] for f in results),upstream_unchanged_files=len(inventory));write_json(output/'P6_manifest.json',result)
    import runpy
    runpy.run_path(str(root/'scripts/finalize_p6.py'),run_name='__main__')
    return output/'stage_report.md'


def write_report(output,result):
    lines=['# P6 Coordination Mechanism Development and freeze','','State: FROZEN_FOR_P7F1/F2 complete development; F3 is G0 PASS, G1 INCONCLUSIVE and has not entered the full branch. outer results is maintained without unblinding.','',
        'The rules are recorded in configs/round2.json and docs/protocol_decisions.md before P6 fit. This is the same as the P4 10k JSD linear CRE. Post-CRE for 12 fixed slot, 7 subset equal weighting and two equal weighting domains; Six of the label correction models used only the evidence that has already been examined.','',
        '| Fold | Methods | cap | Early stop threshold | Final threshold | coverage | risk | Average call | Integration component cost estimate |','|---|---|---:|---|---|---:|---|---:|---:|']
    for fold in result['folds']:
        folder=output/fold['fold'];policies=json.loads((folder/'frozen_policies.json').read_text())
        for name,points in policies.items():
            for p in points:
                s=p['summary'];lines.append(f"| {fold['fold']} | {name} | {p['cap']} | {p['tau_stop']} | {p['tau_final']} | {s['coverage']:.6f} | {s['risk']} | {s['mean_calls']:.6f} | {s['normalized_cost']:.6f} |")
    lines+=['','The prior stop threshold null indicates that the method rule is closed or empty, see early_stop_status in particular; The final threshold null indicates that no feasible is the final acceptance threshold and the remainder records abstention. All candidate coverage, point risk and 2000 blocks of bootstrap unilateral 95% are conserved at cap*_selection.json. The risk range is a point-to-point development of selective evidence.','',
        'The complete fixed cascade sequence and selection, cascade learning, MoE, fusion and route controls have been preserved. budget_matched.json randomly mixes preregistration to select two operating points according to the same call and test cost budgets, reporting the expected quantity and no feasible status.','',
        'R_cal_diagnostics.json preserve the probability of / sequence diagnosis; The model files save target counts, weights, standardization, solver gradients, and time. The single labels model uses the preregistration Laplace constant. recording the trajectory at each step and preserving the inquiries and evidence; frozen_trace retains only the query forefront of the actual strategy, while decisions retains the stop / and finally accepts / abstention status.','',
        '## equipment and time consuming','',f" GPU:RTX A5000 CUDA:0; Recorded running time {result['seconds']:.3f} Seconds.",'',
        '| fold | time consuming (s) | peak showing (MiB) |','|---|---:|---:|']
    for f in result['folds']:lines.append(f"| {f['fold']} | {f['seconds']:.3f} | {f['gpu_memory']['peak_used_mib']:.1f} |")
    if 'gpu_peak_used_mib' in result:lines+=['',f" GPU peak occupancy has been recorded throughout the phase {result['gpu_peak_used_mib']:.1f} MiB, using the CUDA total sample size minus the available 100ms sample size. The peak in the F1 table comes from the development run after recycling of fit products; The peak of the process in the first round of F1 fit is 4291.5 MiB.",'']
    if 'timing_note' in result:lines+=['','Time scale: the total is the recorded time limit. The F2 verification run prior to the interruption is maintained by the progress log for 901.8 seconds, with the 94 item having completed its trajectory separately without discharging; The previous F1 check and the current run are scheduled respectively. Prior to the interruption, F2 checked the peak 4941.5 MiB. The scrapped latency measurement attempts were kept separately in logs, not counting the running time of the effective.','']
    lines+=['','latency.json preserve the 30 preheating of each model, the entire sample of 1000 single flow measurements, mean / median /p95 and cache readings and batch throughput; coordinator_latency.json preserves the path to merge Post-CRE to actually call latency. The cost table uses a single flow query chain to unify median.','',
        f"{result['upstream_unchanged_files']} The size of an upstream file is consistent with the time it was modified in nanoseconds. visibility_validation.json verifies the consistency of the drug without questioning the evidence; P6_validation.json verifies the number of fit rows, strategic thresholds and range of calls.",'',
        'The products are: outputs/round2_v2/P6/{F1,F2}/models, policies, frozen_policies.json, budget_matched.json, R_cal_diagnostics.json, latency.json, manifest.json; F3_branch.json; round2_rules.json,execution_v2.json,P6_manifest.json,P6_validation.json.','',
        'P6 development is completed and freeze and P7 external evaluation remains to be carried out.','']
    path=output/'stage_report.md';path.write_text('\n'.join(lines));return path
