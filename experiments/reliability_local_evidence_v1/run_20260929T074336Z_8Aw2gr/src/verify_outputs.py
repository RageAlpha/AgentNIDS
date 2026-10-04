"""Required row, denominator and count checks on this run only."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from core import readj,writej,COUNT_KEYS,METHODS,now
run=Path(sys.argv[1]);p=readj(run/'config/protocol_v1.json');f=readj(run/'calibration/frozen_rules.json');checks=readj(run/'checks/basic_checks.json')
c=pd.read_csv(run/'tables/comparison.csv');s=pd.read_csv(run/'tables/segment_metrics.csv');w=pd.read_csv(run/'tables/window_counts.csv');local=pd.read_csv(run/'tables/local_signal.csv')
for domain,dp in p['domains'].items():
    sample=np.load(run/f'pilot/{domain}_evaluation_sample_ranks.npy');nw=len(sample)//100
    assert sample.shape==(dp['expected_pilot_rows'],)
    assert np.all(np.diff(sample)>0) and np.all((sample>=dp['evaluation'][0])&(sample<dp['evaluation'][1]))
    assert np.array_equal((sample.reshape(nw,100)-dp['evaluation'][0])//10000,np.repeat(np.arange(nw)[:,None],100,axis=1))
    for method in METHODS:
        total=c[(c.domain==domain)&(c.method==method)].iloc[0]
        sw=w[(w.domain==domain)&(w.method==method)];ss=s[(s.domain==domain)&(s.method==method)]
        assert len(sw)==nw and len(ss)==4
        for key in COUNT_KEYS:
            assert int(sw[key].sum())==int(ss[key].sum())==int(total[key])
        for risk,wrong,den in [('R_B','FN','accepted_B'),('R_A','FP','accepted_A')]:
            if total[den]==0: assert pd.isna(total[risk])
            else: assert np.isclose(total[risk],total[wrong]/total[den])
    assert len(local[local.domain==domain])==56
    completion=readj(run/f'pilot/{domain}_completed.json');assert f['freeze_time']<completion['completed_time']
    checks['domains'][domain]['final_saved_table_checks']=dict(sample_range_unique_and_100_per_window=True,all_15_count_columns_additive=True,zero_denominators_NA=True,all_56_signal_strata_saved=True)
assert int(c[c.method=='RAW'].N.sum())==442500
checks['final_verification_time']=now();writej(run/'checks/basic_checks.json',checks)
print('PASS: 442500 sampled rows; 100/window; 112 signal strata; saved count sums and risk denominators.')
