from audit_common import *
P=read(PILOT/'protocol.json');w=np.load(OUT/'ToN_independent_window_counts.npz')['counts'].sum(axis=1);assert w.shape==(2752,2,2,4)
bs=np.stack([w[start:start+10].sum(0) for start in range(0,2752,10)]);assert bs.shape==(276,2,2,4) and bs[-1].sum()==20000
oldb=np.load(PILOT/'bootstrap_block_counts.npz')['counts'];assert np.array_equal(bs,oldb)
oldranges=rows(PILOT/'bootstrap_blocks.csv')
for i,r in enumerate(oldranges):assert [int(r[k]) for k in ['block_id','first_window_index','last_window_index_exclusive','windows','rank_start','rank_end_exclusive','rows']]==[i,i*10,min(i*10+10,2752),min(10,2752-i*10),i*100000,min(i*100000+100000,27520000),int(bs[i].sum())]
idx=np.load(PILOT/'bootstrap_indices.npy');mult=np.load(PILOT/'bootstrap_multiplicity.npy');ref=np.random.Generator(np.random.PCG64(20260915)).integers(0,276,(2000,276),dtype=np.int64);assert np.array_equal(idx,ref)
metrics=[r['metric'] for r in rows(PILOT/'bootstrap_intervals.csv')];savedreps=rows(PILOT/'bootstrap_replicate_metrics.csv');values={k:[] for k in metrics};checked=0;diff=[]
for i,row in enumerate(idx):
 assert np.array_equal(np.bincount(row,minlength=276),mult[i])
 t=bs[row].sum(0);m=reference_metrics(t)
 for k in metrics:
  v=m[k];values[k].append(np.nan if v is None else v)
  for key in [k,k+'_numerator',k+'_denominator']:
   if key in m and key in savedreps[i]:
    expected=savedreps[i][key];actual=m[key];good=expected=='NA' if actual is None else float(expected)==actual;checked+=1
    if not good:diff.append(dict(replicate=i,field=key,old=expected,new=actual))
for r in rows(PILOT/'bootstrap_intervals.csv'):
 vals=np.array(values[r['metric']]);finite=vals[np.isfinite(vals)];assert int(r['valid_replicates'])==len(finite)
 if len(finite):assert np.array_equal(np.quantile(finite,[.025,.975],method='linear'),[float(r['lower_95']),float(r['upper_95'])])
 else:assert r['lower_95']==r['upper_95']=='NA'
assert not diff
empty=reference_metrics(np.zeros((2,2,4),dtype='i8'));assert empty['selective_risk'] is None and empty['prediction_set_coverage'] is None
allreject=np.zeros((2,2,4),dtype='i8');allreject[1,1,0]=10;assert reference_metrics(allreject)['selective_risk'] is None
write('statistics_validation.json',dict(status='PASS',replicates=2000,blocks=276,block_windows=10,last_block_windows=2,regenerated_indices_match=True,multiplicity_matches=True,independent_replay_blocks_match=True,replicate_fields_checked=checked,ratios_recomputed_from_counts=True,paired_gap_intervals_recomputed=True,percentile_linear_CIs_exactly_match=True,zero_denominator_NA_test=True,source_bootstrap_not_used=True,scope='ToN overall only',limitations=['Fixed 10-window blocks do not prove dependence length adequately captured','Time-segment rates differ; stationary/exchangeable target resampling population not established','Conditional on frozen model/q and observed time sequence; excludes calibration uncertainty','TP bootstrap [0,0] follows observed zero TP in every block, not proof future detection probability zero','731 accepted samples limit selective-population inference but do not invalidate observed 27.52M-row near-total rejection']))
print('PASS independent bootstrap audit',checked,'replicate fields',flush=True)
