from common import *
from implementation import scores,restore_q
# Verify fixed q, do not sort/select/recompute a threshold.
p=read(OUT/'protocol.json');f=read(OUT/'q_freeze.json');q=restore_q(f['q_serialized'])
cache=np.load(p['inputs']['UNSW']['probability_cache'],mmap_mode='r');lab=np.load(p['inputs']['UNSW']['labels'],mmap_mode='r')
y=lab[1420000:1780000];raw=cache['raw_probability'][1420000:1780000];s=scores(raw,y)
assert hashlib.sha256(s.tobytes()).hexdigest()==f['score_sha256']
assert int((s<q).sum())==f['scores_less_than_q'] and int((s==q).sum())==f['scores_equal_q'] and int((s>q).sum())==f['scores_greater_than_q']
rs=[]
for v,name in [(0,'Benign'),(1,'Attack')]:
 ss=s[y==v];rs.append(dict(true_class=name,rows=len(ss),below_q=int((ss<q).sum()),equal_q=int((ss==q).sum()),above_q=int((ss>q).sum()),included_at_frozen_q=int((ss<=q).sum()),empirical_calibration_coverage=float((ss<=q).mean())))
csvwrite('calibration_class_summary.csv',rs)
write('calibration_verification.json',dict(status='PASS',q_not_reselected=True,sorted_or_selected_again=False,score_hash_and_tie_counts_match=True,calibration_rows=360000,q_sha256=sha(OUT/'q_freeze.json'),code_sha256=sha(OUT/'calibration_audit.py')))
print(rs)
