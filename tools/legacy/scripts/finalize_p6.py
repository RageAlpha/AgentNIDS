import json
import time
import sys
import re
from pathlib import Path
from datetime import datetime,timezone
import cupy as cp
import numpy as np
from nids.data.partitions import PreparedData
from nids.coordination.evidence import Development
from nids.coordination.replay import trajectory
from nids.coordination.validation import check_visibility,validate_saved
from nids.coordination.diagnostics import model_diagnostics,query_diagnostics
from nids.coordination.matched import evaluate_randomized
from nids.p6 import write_report,freeze_trace
from nids.coordination.report import augment
from nids.p4 import release
from nids.models.base_models import GpuMemoryMonitor
from nids.replay.predictions import write_json

verification_started=time.perf_counter()
root=Path('/home/karri/projects/NIDSProject_Professor_Run');c=json.loads((root/'configs/execution_v2.json').read_text());prepared=PreparedData(Path(c['prepared_run']));output=root/c['outputs']['round2']/'P6';actual=[]
resume='--resume-interrupted' in sys.argv
freeze_verified='--freeze-verified' in sys.argv
completed=set()
prior_verification_seconds=0.
if resume:
 actual=json.loads((output/'F1/actual_replay_validation.json').read_text())
 prior_verification_seconds=json.loads((output/'F1/verification_runtime.json').read_text())['seconds']+901.8
 completed=set(re.findall(r'F2/([^/]+)/cap(\d)/([^:]+): actual conditional queries PASS',(root/'logs/P6.log').read_text()))
if freeze_verified:
 actual=[r for fold in ['F1','F2'] for r in json.loads((output/fold/'actual_replay_validation.json').read_text())]
 prior_verification_seconds=901.8+sum(json.loads((output/fold/'verification_runtime.json').read_text())['seconds'] for fold in ['F1','F2'])+json.loads((output/'threshold_detail_restoration.json').read_text())['seconds']
for fold in ['F1','F2']:
 if freeze_verified:continue
 if resume and fold=='F1':continue
 fold_started=time.perf_counter();monitor=GpuMemoryMonitor(0);monitor.start()
 folder=output/fold;fitted=json.loads((folder/'models/fitted.json').read_text());initial=json.loads((folder/'initial_detector.json').read_text());latency=json.loads((folder/'cost_model.json').read_text());policies=json.loads((folder/'frozen_policies.json').read_text())
 write_json(folder/'visibility_validation.json',check_visibility(fitted,latency['normalized_cost']))
 cal=[Development(c,prepared,fold,d,'R_cal') for d in c['folds'][fold]['development']]
 if not resume:model_diagnostics(cal,fitted,c,folder)
 release()
 data=[Development(c,prepared,fold,d,'R_select') for d in c['folds'][fold]['development']]
 names={'learned_cascade':'learned','soft_moe':'soft','hard_moe':'hard','best_single':'best'}
 for name,points in policies.items():
  kind='fixed' if name.startswith('fixed_') else names.get(name,name);order=list(map(int,name.split('_')[1])) if kind=='fixed' else None
  policy=folder/'policies'/name
  for point in points:
   cap=point['cap']
   for d in data:
    decision_path=policy/f'{d.domain}_cap{cap}_decisions.npz';stored=np.load(decision_path);stage=stored['stage'].copy();accepted=stored['accepted'].copy();early=stored['early_stop'].copy();stored.close()
    np.savez(decision_path,stage=stage,accepted=accepted,early_stop=early,reason=np.where(early,0,np.where(accepted,1,2 if point['tau_final'] is not None else 3)).astype(np.uint8))
    path=policy/f'{d.domain}_cap{cap}_actual_query_trace.npy'
    reused=resume and (name,str(cap),d.domain) in completed
    if reused:
     r=dict(rows=d.n,seconds=None,visibility='query_only_accessor',cap=cap,tau_stop=point['tau_stop'],path=str(path),reused_completed_trace=True,timing='per_trace_timing_not_persisted_before_interruption')
    else:r=trajectory(d,fitted,latency,initial,kind,path,order,cap=cap,tau_stop=point['tau_stop'])
    actual_trace=np.load(path,mmap_mode='r');frozen=np.load(policy/f'{d.domain}_cap{cap}_frozen_trace.npy',mmap_mode='r');decisions=np.load(policy/f'{d.domain}_cap{cap}_decisions.npz')
    for a in range(0,d.n,262144):
     b=min(a+262144,d.n);np.testing.assert_array_equal(actual_trace['query'][a:b],frozen['query'][a:b]);stage=decisions['stage'][a:b];row=np.arange(b-a)
     np.testing.assert_allclose(actual_trace['fused'][a:b][row,stage],frozen['fused'][a:b][row,stage],rtol=0,atol=1e-14)
    reads=int((actual_trace['query']>=0).sum())
    if reused:r['detector_record_reads']=reads
    assert r['detector_record_reads']==reads;r.update(fold=fold,method=name,domain=d.domain,status='PASS');actual.append(r)
    write_json(folder/'actual_replay_checkpoint.json',[r for r in actual if r['fold']==fold])
    print(f'{fold}/{name}/cap{cap}/{d.domain}: actual conditional queries PASS',flush=True);release()
 query_diagnostics(data,folder,policies)
 evaluate_randomized(data,folder,c['seed'])
 write_json(folder/'actual_replay_validation.json',[r for r in actual if r['fold']==fold])
 write_json(folder/'verification_runtime.json',dict(seconds=time.perf_counter()-fold_started,gpu_memory=monitor.stop()))
# Every upstream artifact remains frozen.
for item in json.loads((output/'upstream_inventory.json').read_text())['files']:
 stat=Path(item['path']).stat();assert stat.st_size==item['bytes'] and stat.st_mtime_ns==item['mtime_ns']
validation=validate_saved(output);assert len(actual)==192
validation.update(actual_conditional_replays=len(actual),future_and_unqueried_invariance=True)
if freeze_verified:
 identity=json.loads((output/'identity_validation.json').read_text());restored=json.loads((output/'threshold_detail_restoration.json').read_text());assert identity['status']=='PASS' and restored['status']=='PASS'
 validation.update(all_record_identity_and_causal_predecessors=True,identity_manifests=len(identity['checks']),restored_threshold_details=len(restored['checks']))
write_json(output/'P6_validation.json',validation)
result=json.loads((output/'P6_manifest.json').read_text());result['status']='FROZEN_FOR_P7';result['frozen_at_utc']=datetime.now(timezone.utc).isoformat();result['actual_replay_seconds']=sum(r['seconds'] for r in actual if r['seconds'] is not None);result['development_seconds']=result['seconds'];result['verification_seconds']=prior_verification_seconds+time.perf_counter()-verification_started;result['seconds']+=result['verification_seconds']
if resume or freeze_verified:result['timing_note']='Total is a recorded lower bound: interrupted F2 verification retained 901.8 seconds from the progress log; 94 completed F2 per-trace durations were not persisted. Superseded timing attempt excluded. F2 interrupted peak 4941.5 MiB. Threshold detail restoration included.'
if freeze_verified:result['gpu_peak_used_mib']=max(4941.5,restored['gpu_memory']['peak_used_mib'],*[f['gpu_memory']['peak_used_mib'] for f in result['folds']])
write_json(output/'P6_manifest.json',result)
rules=json.loads((output/'round2_rules.json').read_text());rules.update(status='ALL_P6_ARTIFACTS_FROZEN_FOR_P7',frozen_at_utc=result['frozen_at_utc'],fold_artifacts={fold:str(output/fold/'frozen_policies.json') for fold in ['F1','F2']},F3='NOT_ELIGIBLE');write_json(output/'round2_frozen.json',rules)
print(write_report(output,result))
augment(output,result)
