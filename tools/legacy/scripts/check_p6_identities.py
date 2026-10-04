"""Check every P6 development identity against frozen sorting and windows."""
import json,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
root=Path('/home/karri/projects/NIDSProject_Professor_Run');c=json.loads((root/'configs/execution_v2.json').read_text());out=root/c['outputs']['round2']/'P6';started=time.perf_counter();checks=[]
manifest=json.loads((out/'P6_manifest.json').read_text())
for fold in manifest['folds']:
 for part,domains in fold['audits'].items():
  for audit in domains:
   domain=audit['domain'];assert domain in c['folds'][fold['fold']]['development']
   order=np.load(Path(c['prepared_run'])/f'{domain}_sorted_original_rows.npy',mmap_mode='r');ends=np.load(Path(c['prepared_run'])/f'{domain}_sorted_end_ms.npy',mmap_mode='r')
   all_windows=pq.read_table(root/c['outputs']['shift']/'10k'/f'{domain}_windows.parquet').to_pylist();lookup={w['window_id']:w for w in all_windows}
   windows=sorted([w for w in all_windows if w['scheme']=='development' and w['partition']==part],key=lambda w:w['rank_start'])
   for w in windows:
    assert w['rank_end_exclusive']-w['rank_start']==10000
    if w['initialization']:assert w['shift']==0 and w['previous_window_id'] is None
    else:
     previous=lookup[w['previous_window_id']];assert previous['stream_id']==w['stream_id'];assert previous['last_available_ms']<=w['first_available_ms'];assert previous['rank_end_exclusive']<=w['rank_start'];assert w['shift']==previous['jsd']
   ids=pa.array([w['window_id'] for w in windows]);prev=pa.array([w['previous_window_id'] for w in windows],type=pa.string());init=np.array([w['initialization'] for w in windows]);position=audit['rank_start'];path=out/fold['fold']/'identities'/f'{domain}_{part}.parquet'
   for batch in pq.ParquetFile(path).iter_batches(batch_size=262144):
    table=pa.Table.from_batches([batch]);n=len(table);rank=table['rank'].to_numpy();np.testing.assert_array_equal(rank,np.arange(position,position+n));np.testing.assert_array_equal(table['original_row'].to_numpy(),order[rank]);np.testing.assert_array_equal(table['available_ms'].to_numpy(),ends[rank]);assert table['complete_window'].to_numpy().all()
    expected_id=pc.binary_join_element_wise(pa.scalar(domain+':'),pc.cast(table['original_row'],pa.string()),pa.scalar(''));assert bool(pc.all(pc.equal(table['record_id'],expected_id)).as_py())
    index=(rank-audit['rank_start'])//10000
    assert table['window_id'].combine_chunks().equals(pc.take(ids,pa.array(index)))
    assert table['previous_window_id'].combine_chunks().equals(pc.take(prev,pa.array(index)))
    np.testing.assert_array_equal(table['initialization'].to_numpy(),init[index]);position+=n
   assert position==audit['rank_end_exclusive'];checks.append(dict(fold=fold['fold'],domain=domain,partition=part,rows=audit['rows'],windows=len(windows),status='PASS'));print(checks[-1],flush=True)
(out/'identity_validation.json').write_text(json.dumps(dict(status='PASS',checks=checks,seconds=time.perf_counter()-started,device='CPU_Parquet_and_identity_validation',all_record_identity_and_causal_predecessors=True),indent=2)+'\n')
