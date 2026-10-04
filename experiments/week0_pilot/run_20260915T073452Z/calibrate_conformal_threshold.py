from common import *
from implementation import rf_p,scores,threshold,serialize_q,restore_q
assert not (OUT/'q_freeze.json').exists(),'q already frozen; refuse recalibration'
f=read(OUT/'protocol_freeze.json');assert sha(OUT/'protocol.json')==f['protocol_sha256']
assert sha(OUT/'implementation.py')==f['implementation_sha256']
assert read(OUT/'implementation_check.json')['status']=='PASS'
protocol=read(OUT/'protocol.json');cal=protocol['calibration'];a,b=cal['rank_start'],cal['rank_end_exclusive']
cache=np.load(protocol['inputs']['UNSW']['probability_cache'],mmap_mode='r')
labels=np.load(protocol['inputs']['UNSW']['labels'],mmap_mode='r')
original=np.load(protocol['inputs']['UNSW']['original_rows'],mmap_mode='r')
p=np.array(cache['raw_probability'][a:b]);y=np.array(labels[a:b]);rf_p(p)
assert p.dtype==np.float64 and np.isin(y,[0,1]).all()
assert len(p)==360000 and np.bincount(y,minlength=2).tolist()==[325600,34400]
assert hashlib.sha256(p.tobytes()).hexdigest()==read(PREF/'calibration_data_check.json')['raw_field_sha256']
started=now();s=scores(p,y);q,k=threshold(s,'1/10');assert k==324001
obj=serialize_q(q)
assert restore_q(obj)==q and (math.isinf(q) or restore_q(obj).hex()==q.hex())
summary=dict(n=len(p),epsilon={'numerator':1,'denominator':10},k_one_based=k,k_zero_based=k-1,q_decimal=repr(q) if math.isfinite(q) else None,q_serialized=obj,score_min=float(s.min()),score_max=float(s.max()),score_mean=float(s.mean()),score_unique_count=int(len(np.unique(s))),scores_less_than_q=int((s<q).sum()),scores_equal_q=int((s==q).sum()),scores_greater_than_q=int((s>q).sum()),benign=325600,attack=34400,rank_start=a,rank_end_exclusive=b,probability_sha256=hashlib.sha256(p.tobytes()).hexdigest(),label_sha256=hashlib.sha256(y.tobytes()).hexdigest(),identity_slice_sha256=hashlib.sha256(original[a:b].tobytes()).hexdigest(),score_sha256=hashlib.sha256(s.tobytes()).hexdigest(),score_array_saved=False,computation_started_at_utc=started)
assert summary['scores_less_than_q']<k<=summary['scores_less_than_q']+summary['scores_equal_q']
write('calibration_summary.json',summary)
freeze=dict(version='week0-q-freeze-v1.0',frozen_at_utc=now(),**summary,calibration_summary_sha256=sha(OUT/'calibration_summary.json'),protocol_sha256=f['protocol_sha256'],input_sources_sha256=sha(OUT/'source_hashes.csv'),code_sha256={n:sha(OUT/n) for n in ['calibrate.py','implementation.py','common.py']},selection_calls_on_real_calibration=1,source_calibration_only=True)
with (OUT/'q_freeze.json').open('x') as h:json.dump(freeze,h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
assert restore_q(read(OUT/'q_freeze.json')['q_serialized'])==q
print(json.dumps(summary,indent=2));print('q frozen before evaluation:',freeze['frozen_at_utc'])
