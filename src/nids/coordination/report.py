import csv
import json
from pathlib import Path


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def augment(output,result):
    lines=[];fit_rows=[];latency_rows=[];policies=[];budgets=[];diagnostics=[];g2=[]
    lines+=['','## Data scope and phase boundaries','','| Fold | Field | partition | Full record | complete windows | Initial record | Excluding tail Record |','|---|---|---|---:|---:|---:|---:|']
    for fold in result['folds']:
        for part,domains in fold['audits'].items():
            for d in domains:lines.append(f"| {fold['fold']} | {d['domain']} | {part} | {d['rows']} | {d['windows']} | {d['initialization_rows']} | {d['tail']} |")
    lines+=['','## Feasibility of the main coordination strategies','','| Fold | cap | Predetermined Threshold | Predetermined risk upper bound | Predetermined Threshold | Predetermined risk upper bound Predetermined | Predetermined | Predetermined |','|---|---:|---|---|---|---|---|---|']
    for fold in ['F1','F2']:
        points=json.loads((output/fold/'frozen_policies.json').read_text())['coordinator']
        for p in points:
            p=json.loads((output/fold/'policies/coordinator'/f"cap{p['cap']}_selection.json").read_text())
            upper=[]
            for key,candidates in [('tau_stop','stop_candidates'),('tau_final','final_candidates')]:
                upper.append(next((v['upper95'] for v in p[candidates] if v['threshold']==p[key]),None))
            lines.append(f"| {fold} | {p['cap']} | {p['tau_stop']} | {upper[0]} | {p['tau_final']} | {upper[1]} | {p['early_stop_status']} | {p['final_status']} |")
    lines+=['','F2 All budget matching candidates are NO_FEASIBLE_MATCHED_OPERATING_POINT; The no feasible result is maintained according to the freeze rule. The G0/G1 qualification judgment of P4 maintains the original conclusion, and the P6 feasibility is reported separately.','']
    lines+=['','## compares initial detector to strong baseline','']
    detector_names=['RF Traffic17','XGBoost Protocol17','MLP Behavior15']
    for fold in ['F1','F2']:
        initial=json.loads((output/fold/'initial_detector.json').read_text());lines.append(f"{fold} The initial detector is {detector_names[initial['selected']]} The equal weighting error rate is selected according to R_cal in both domains.")
    points=json.loads((output/'F1/frozen_policies.json').read_text());fixed=json.loads((output/'F1/fixed_cascade_selection.json').read_text())['3']
    lines+=['','| F1 Monitoring | cap | coverage | risk | Average calling of | components cost estimate of |','|---|---:|---:|---|---:|---:|']
    for name in ['coordinator',fixed,'soft_moe','hard_moe','learned_cascade']:
        p=points[name][-1];m=p['summary'];lines.append(f"| {name} | {p['cap']} | {m['coverage']:.6f} | {m['risk']} | {m['mean_calls']:.6f} | {m['normalized_cost']:.6f} |")
    lines+=['','The complete cost and budget matching results are preserved, respectively, and the P6 development results have not yet concluded that the main coordination strategy is superior to the strong baseline. F2 All methods retain the no feasible acceptance and budget matching results.','']
    for fold in ['F1','F2']:
        folder=output/fold
        for path in (folder/'models').glob('*.json'):
            if path.name=='fitted.json':continue
            m=json.loads(path.read_text());fit_rows.append(dict(fold=fold,model=m['name'],rows=m['rows'],weight_sum=m['weight_sum'],weighted_positive=m['weighted_positive'],status=m['status'],iterations=m.get('iterations'),gradient_inf=m.get('gradient_inf'),objective=m.get('objective'),seconds=m['seconds'],features=','.join(m['features'])))
        latency=json.loads((folder/'latency.json').read_text())
        for mid,m in latency['models'].items():latency_rows.append(dict(fold=fold,detector=mid,median_ms=m['median_ms'],mean_ms=m['mean_ms'],p95_ms=m['p95_ms'],batch_flows_per_second=m['batch_detector_flows_per_second']))
        for method,points in json.loads((folder/'frozen_policies.json').read_text()).items():
            for p in points:policies.append(dict(fold=fold,method=method,cap=p['cap'],tau_stop=p['tau_stop'],tau_final=p['tau_final'],early_stop_status=p['early_stop_status'],final_status=p['final_status'],**p['summary']))
        for m in json.loads((folder/'budget_matched.json').read_text()):budgets.append(dict(fold=fold,method=m['method'],calls_budget=m['calls_budget'],cost_budget=m['cost_budget'],status=m['status'],**(m['selection'] or {})))
        for path in folder.glob('*_model_R_cal_diagnostics.json'):
            domain=path.name.split('_')[0]
            for name,m in json.loads(path.read_text()).items():diagnostics.append(dict(fold=fold,domain=domain,model=name,**{k:v for k,v in m.items() if not isinstance(v,(list,dict))}))
        for method,entry in json.loads((folder/'G2_query_diagnostics.json').read_text()).items():
            for i in range(2):
                mass={key:sum(d['steps'][i][key]/d['rows'] for d in entry['domains'])/2 for key in ['consulted','corrective_hits','oracle_opportunities']}
                g2.append(dict(fold=fold,method=method,step=i+2,**mass,corrective_rate=mass['corrective_hits']/mass['consulted'] if mass['consulted'] else None,oracle_recovery=mass['corrective_hits']/mass['oracle_opportunities'] if mass['oracle_opportunities'] else None))
    for name,rows in [('fit_summary',fit_rows),('latency_summary',latency_rows),('policy_summary',policies),('budget_summary',budgets),('model_R_cal_metrics',diagnostics),('G2_summary',g2)]:write_csv(output/f'{name}.csv',rows)
    lines+=['','## main coordination strategy phased label correction diagnosis','','Calculation of the equal weighting mass of the two fields that have been fully recorded by consultation and hit, followed by a conditional ratio; For full contrast, see G2_summary.csv.','',
        '| discount | cap | query step | consult quality | conditions label correction rate | recovery opportunity hit rate |','|---|---:|---:|---:|---|---|']
    for m in g2:
        if m['method']=='coordinator/cap3':lines.append(f"| {m['fold']} | 3 | {m['step']} | {m['consulted']:.6f} | {m['corrective_rate']} | {m['oracle_recovery']} |")
    lines+=['','## fit and the real question latency','',
        '| folded | model | training number | state | replacement | gradient∞ parameter | time consuming (s) |','|---|---|---:|---|---:|---|---:|']
    for m in fit_rows:lines.append(f"| {m['fold']} | {m['model']} | {m['rows']} | {m['status']} | {m['iterations']} | {m['gradient_inf']} | {m['seconds']:.3f} |")
    lines+=['','| Folded | detector | Single flow median (ms) | mean (ms) | p95(ms) | detector Batch ingested (flow/s) |','|---|---|---:|---:|---:|---:|']
    for m in latency_rows:lines.append(f"| {m['fold']} | {m['detector']} | {m['median_ms']:.6f} | {m['mean_ms']:.6f} | {m['p95_ms']:.6f} | {m['batch_flows_per_second']:.3f} |")
    lines+=['','All cached reading timings are saved separately from the actual query chain timings. The table summarizes the estimates of the complex cost as test components, including Pre-CRE, pathways, fusion /Post-CRE and historical JSD deployments; The mass throughput is calculated as a separate detector calibre. method_overhead_latency.json preserves 1000 testing at various stages, cost_model.json preserves unification and deployment.','',
        '## budget matching with query diagnosis','',
        'The budget_summary.csv complete list of unified call numbers with a possible state under the unified cost budget and a random mix of freeze. budget_mixture_realized.json also reports the same freeze strategy for actual calls in the development recycle, cost, coverage/risk; Random limited samples of achievement values and expected budgets are shown. The choose_a array is positioned to the actual query trajectory of the corresponding cap.','',
        'G2_query_diagnostics.json gradually records the consultation number, actual label correction lifetime number and oracle recovery opportunities; oracle only reads caching with labels after the strategic and actual query trajectory of freeze. R_cal all label correction, gate control and seven subset Post-CRE diagnoses see model_R_cal_metrics.csv and complete JSON in each domain. The six candidates with fixed cascade are preserved in full selective order as a coordinated strategy for controlling and withdrawing the preregistration candidate.','',
        '## record verification by record','',
        'identities/*.parquet keeps record_id, rank, original_row, window_id, previous_window_id, availability with the initialization tag, and all tactics refer to the same complete windows record. actual_query_trace performs cap and tau_stop condition queries through freeze; The recording of the query sequence and the probability of eventual convergence is consistent with the development of product selection reviews. After stopping, do not read the candidate cache.','',
        'identity_validation.json step by step verifies 12 development of identity manifest with both sequencing, availability time, window and front drive; If the shift of a non-initialization window is equal to the same flow, the front drive window JSD has been completed, and the initialization of shift is 0. Parquet and identity are checked using CPU, model fit, pathway, fusion and threshold bootstrap using CUDA.','',
        'decisions.npz’s reason code: 0= is pre-accepted, 1= is ultimately accepted, 2= is below the final threshold abstention and 3= is no feasible and abstention is the final threshold. freeze rules and all methods refer to round2_frozen.json.','',
        f" Development of fit / is time consuming {result.get('development_seconds',result['seconds']):.3f} Seconds; Additional diagnostics and actual conditions recharge time {result.get('verification_seconds',0):.3f} Seconds. Both parts actually use CUDA, and the verification phase shows the storage and time consuming verification_runtime.json.",'',
        'After reuse of full path candidate relay for threshold search, the relay module achieves freeze threshold-driven condition queries and recalculates the actual strategy trajectory; The model and threshold values are consistent. External forecasts and labels ratings remain at P7.','',
        'The reasoning module implements the P4 call as the current h,u,s linear CRE required, and fixes the parameters permanently to GPU. A inference_fix_validation.json record is checked step by step to match the probability of the original P4; After repairs, re-measurement of latency and completion of the strategy selection, the completed F1 fit continues to be reused.','',
        'At the end of the synchronized status description, the threshold candidate is briefly covered. Using the selection modules and the original freeze repositioning, all 96 group candidates are reconciled for each confirmation threshold, summarizing indicator and each recording stage, acceptance, early termination and cause coding. threshold_detail_restoration.json preserves verification and testing time, and P6_validation.json checks the complete preservation of each group of 101 candidates in advance and in the end.','',
        'The code is src/nids/p6.py, src/nids/coordination/{glm,evidence,fitting,latency,replay,selection,validation,diagnostics,matched,report}.py; Unified input src/nids/run.py; Run with the end scripts/run_p6.sh, scripts/finalize_p6.py; The protocol checks tests/check_p6_protocol.py; freeze registers configs/round2.json, docs/protocol_decisions.md.','']
    path=output/'stage_report.md';path.write_text(path.read_text()+'\n'.join(lines))
