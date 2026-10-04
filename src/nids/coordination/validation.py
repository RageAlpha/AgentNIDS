import json
from pathlib import Path
import cupy as cp
import numpy as np
from .evidence import post_features
from .replay import route
from .selection import choose,bounds


def check_visibility(fitted,cost):
    class Visible:
        def __init__(self):
            self.mask=cp.asarray([[True,False,False],[False,True,False]]);self.p=cp.asarray([[.2,0,0],[0,.8,0]]);self.u=cp.asarray([[.4,0,0],[0,.3,0]]);self.h=cp.asarray([[.8,0,0],[0,.9,0]]);self.r=self.h.copy();self.s=cp.asarray([.1,.2])
        def current(self,rows,current):return cp.column_stack((self.h[rows,current],self.u[rows,current],self.s[rows],self.p[rows,current]>=.5))
    v=Visible();rows=cp.arange(2);current=cp.asarray([0,1]);random=cp.full((2,2),.4)
    for kind in ['coordinator','learned','uncertainty','hard','cheapest','random','fixed']:
        for x in [v.p,v.u,v.h,v.r]:x[~v.mask]=0
        before=route(v,rows,current,kind,fitted,cost,0,[0,1,2],random);post=post_features(v.mask,v.p,v.r,v.s)
        for x in [v.p,v.u,v.h,v.r]:x[~v.mask]=cp.nan
        after=route(v,rows,current,kind,fitted,cost,0,[0,1,2],random)
        np.testing.assert_array_equal(cp.asnumpy(before[0]),cp.asnumpy(after[0]));np.testing.assert_array_equal(cp.asnumpy(before[1]),cp.asnumpy(after[1]));np.testing.assert_array_equal(cp.asnumpy(post),cp.asnumpy(post_features(v.mask,v.p,v.r,v.s)))
    before=route(v,rows,current,'coordinator',fitted,cost,0,[0,1,2],random)
    v.s[1]=.99;v.p[1,1]=.01;v.h[1,1]=.1;v.u[1,1]=.99
    after=route(v,rows,current,'coordinator',fitted,cost,0,[0,1,2],random)
    np.testing.assert_array_equal(cp.asnumpy(before[1][0]),cp.asnumpy(after[1][0]))
    assert choose([dict(coverage=.5,risk=.1,upper95=.11,threshold=.9)]) is None
    assert choose([dict(coverage=.5,risk=.02,upper95=.03,threshold=.8),dict(coverage=.5,risk=.02,upper95=.03,threshold=.9)])['threshold']==.9
    return dict(status='PASS',unqueried_nan_poison_invariance=True,future_row_change_preserves_earlier_route=True,post_mask_invariance=True,threshold_empty_and_tie_rules=True,device='cuda:0')


def validate_saved(output):
    reports=[];models=0
    for folder in sorted(output.glob('F[12]')):
        fits=json.loads((folder/'models/fitted.json').read_text())
        assert fits['post']['rows']==7*sum(a['rows'] for a in json.loads((folder/'fit_audits.json').read_text())['domains'])
        for path in (folder/'models').glob('*.json'):
            if path.name=='fitted.json':continue
            m=json.loads(path.read_text());assert m['status'] in ['CONVERGED','CONSTANT_SINGLE_LABEL']
            if m['status']=='CONVERGED':assert m['gradient_inf']<=1e-4
            models+=1
        for path in (folder/'policies').glob('*/cap*_selection.json'):
            m=json.loads(path.read_text());assert m['summary']['mean_calls']<=3
            assert [r['threshold'] for r in m['stop_candidates']]==[k/100 for k in range(101)]
            assert [r['threshold'] for r in m['final_candidates']]==[k/100 for k in range(101)]
            for key,t in [('stop_candidates',m['tau_stop']),('final_candidates',m['tau_final'])]:
                if t is not None:
                    selected=[r for r in m[key] if r['threshold']==t][0];assert selected['coverage']>=.1 and selected['upper95']<=.05
            reports.append(str(path))
    assert models==40 and len(reports)==96
    return dict(status='PASS',models=models,solver_status_and_gradient=True,policies=len(reports),validated_files=reports,outer_evaluation=False)
