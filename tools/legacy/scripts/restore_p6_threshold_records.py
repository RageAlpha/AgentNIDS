"""Recompute threshold diagnostics from frozen candidate trajectories."""
import json,time
from pathlib import Path
import numpy as np
from nids.data.partitions import PreparedData
from nids.coordination.evidence import Development
from nids.coordination.selection import draws_for_domains,select_thresholds
from nids.models.base_models import GpuMemoryMonitor
from nids.p4 import release
from nids.replay.predictions import write_json
root=Path('/home/karri/projects/NIDSProject_Professor_Run');c=json.loads((root/'configs/execution_v2.json').read_text());prepared=PreparedData(Path(c['prepared_run']));output=root/c['outputs']['round2']/'P6'
started=time.perf_counter();monitor=GpuMemoryMonitor(0);monitor.start();checks=[]
for fold in ['F1','F2']:
 folder=output/fold;data=[Development(c,prepared,fold,d,'R_select') for d in c['folds'][fold]['development']];draws=draws_for_domains(data,c['seed'])
 for name,points in json.loads((folder/'frozen_policies.json').read_text()).items():
  policy=folder/'policies'/name;paths=[policy/f'{d.domain}_R_select_trajectory.npy' for d in data]
  for old in points:
   cap=old['cap'];before=[]
   for d in data:
    with np.load(policy/f'{d.domain}_cap{cap}_decisions.npz') as z:before.append({k:z[k].copy() for k in ['stage','accepted','early_stop','reason']})
   new=select_thresholds(data,paths,cap,draws,policy,allow_stop=name.startswith('fixed_') or name in ['coordinator','learned_cascade','random','cheapest','uncertainty'])
   assert new['tau_stop']==old['tau_stop'] and new['tau_final']==old['tau_final']
   for k,v in old['summary'].items():
    if v is None:assert new['summary'][k] is None
    else:np.testing.assert_allclose(new['summary'][k],v,rtol=0,atol=1e-14)
   for d,expected in zip(data,before):
    with np.load(policy/f'{d.domain}_cap{cap}_decisions.npz') as z:
     for k,v in expected.items():np.testing.assert_array_equal(z[k],v)
   checks.append(dict(fold=fold,method=name,cap=cap,status='PASS',thresholds_metrics_and_all_record_decisions_unchanged=True));print(f'{fold}/{name}/cap{cap}: threshold detail restored; frozen decisions PASS',flush=True);release()
write_json(output/'threshold_detail_restoration.json',dict(status='PASS',checks=checks,seconds=time.perf_counter()-started,gpu_memory=monitor.stop()))
