"""ToN-only calibration action correction. Reads only original calibration artifacts."""
import argparse,json,sys,shlex
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

def now():return datetime.now(timezone.utc).isoformat()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def ratio(a,b):return a/b if b else None

def counts(y,h):
    tn=int(((h==0)&(y==0)).sum());fn=int(((h==0)&(y==1)).sum());fp=int(((h==1)&(y==0)).sum());tp=int(((h==1)&(y==1)).sum())
    nb=int((y==0).sum());na=int((y==1).sum())
    return dict(N=len(y),N_B=nb,N_A=na,TN=tn,FN=fn,FP=fp,TP=tp,accepted_B=tn+fn,accepted_A=tp+fp,R_B=ratio(fn,tn+fn),R_A=ratio(fp,tp+fp),U_B=tn/nb,U_A=tp/na,U_bal=(tn/nb+tp/na)/2)

def accept(rule,s,support):
    if rule['direction']=='OFF':return np.zeros(len(s),bool)
    return ((s>=rule['threshold']) if rule['direction']=='HIGH' else (s<=rule['threshold']))&(support>=rule['u'])

def candidates(h,y,s,q,grid,cls):
    definitions=[dict(id='OFF_T999_U000',direction='OFF',threshold=None,threshold_index=None,u=0.,u_index=0)]
    for direction in ['HIGH','LOW']:
        for ti,t in enumerate(grid):
            for ui,u in enumerate([0.,.5,.75,1.]):
                definitions.append(dict(id=f'{direction}_T{ti:03d}_U{ui:03d}',direction=direction,threshold=t,threshold_index=ti,u=u,u_index=ui))
    support=q if cls==1 else 1-q
    for rule in definitions:
        m=(h==cls)&accept(rule,s,support);n=int(m.sum());err=int((m&(y!=cls)).sum());off=rule['direction']=='OFF'
        reason=[]
        if not off:
            if (h==cls).sum()<10:reason.append('C_CANDIDATE_LABEL_SUPPORT_LT10')
            if n<10:reason.append('ACCEPTED_LT10')
            if n and err/n>.05:reason.append('RISK_GT5_PERCENT')
        rule.update(accepted=n,errors=err,correct=n-err,risk=ratio(err,n),feasible=not reason,infeasible_reasons=';'.join(reason))
    return definitions

def select(method,h,y,s,q,grid):
    branches={c:candidates(h,y,s,q,grid,i) for i,c in enumerate(['B','A'])};rows=[];best=None
    nb=int((y==0).sum());na=int((y==1).sum());direction={'HIGH':0,'LOW':1,'OFF':2}
    for cls,ls in branches.items():
        rows.extend(dict(method=method,record_type='branch',action=cls,**r) for r in ls)
    for b in branches['B']:
        for a in branches['A']:
            ident='B_'+b['id']+'__A_'+a['id'];feasible=b['feasible'] and a['feasible']
            ub=b['correct']/nb;ua=a['correct']/na
            rows.append(dict(method=method,record_type='pair',id=ident,B_id=b['id'],A_id=a['id'],TN=b['correct'],FN=b['errors'],FP=a['errors'],TP=a['correct'],N_B=nb,N_A=na,U_B=ub,U_A=ua,U_bal=(ub+ua)/2,feasible=feasible))
            if feasible:
                key=(-(b['correct']*na+a['correct']*nb),-min(b['correct']*na,a['correct']*nb),b['errors']+a['errors'],b['u'],a['u'],direction[b['direction']],direction[a['direction']],ident)
                if best is None or key<best[0]:best=(key,b,a,ident)
    _,b,a,ident=best
    decision=np.full(len(h),-1,dtype=int)
    for i,c,rule in [(0,'B',b),(1,'A',a)]:decision[(h==i)&accept(rule,s,q if i else 1-q)]=i
    on=sum(r['direction']!='OFF' for r in [b,a])
    result=dict(id=ident,B=b,A=a,status=['NO_FEASIBLE_POLICY','PARTIAL_ACTION_POLICY','TWO_ACTION_POLICY'][on],candidate_label_counts={'B':int((h==0).sum()),'A':int((h==1).sum())},feasible_enabled_candidates={c:sum(r['feasible'] and r['direction']!='OFF' for r in ls) for c,ls in branches.items()},metrics=counts(y,decision))
    return result,rows,decision

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-run',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args();old=args.source_run.resolve();run=args.run_dir.resolve()
    if (run/'calibration/selected_rules.json').exists():raise RuntimeError('Completed selection exists; refusing overwrite')
    frozen=json.loads((old/'calibration/frozen_rules.json').read_text());oldrule=frozen['domains']['ToN']['rules']['G2'];grid=oldrule['grid']
    protocol=dict(experiment='ton_local_action_correction_v1',registered_at=now(),source_run=str(old),domain='ToN',stage='calibration_only',R=200,C=200,total_existing_labels=400,k=15,distance=frozen['protocol']['evidence'],RF='frozen rf_All49 raw_probability',grid=grid,grid_source='original ToN G2 grid; never recomputed',families={'E0':'raw RF label','E1':'q_A>=0.5','E2':'use local label only on disagreement and max(q_A,1-q_A)>=0.75','E3':'E2 with q_prior from R'},gates={'E0-local':'E0 with true local support','E1-local':'E1 with true local support','E2-local':'E2 with true local support','E3-local':'E3 with true local support','E3-prior':'E3 with constant prior support'},selection=frozen['protocol']['gates'],candidate_support='support for final candidate label; s remains original RF max(p,1-p)',reads='original ToN calibration artifacts only; no evaluation records',statistical_status='further development on reused C, not independent validation')
    write(run/'config/protocol.json',protocol)
    # These are the only sample records read. No feature store or pilot evidence is opened.
    with np.load(old/'calibration/ToN_reference.npz') as z:rr=z['rank'];ry=z['y']
    with np.load(old/'calibration/ToN_selection_evidence.npz') as z:rank=z['rank'];y=z['y'];p=z['raw_p'];ls=z['local_support'];neighbors=z['neighbor_ranks']
    roles=pd.read_csv(old/'calibration/sample_roles.csv');roles=roles[roles.domain=='ToN'].copy()
    assert len(rr)==len(rank)==200 and len(set(rr)&set(rank))==0 and len(set(rr)|set(rank))==400
    for label,ranks,truth in [('R',rr,ry),('C',rank,y)]:
        records=roles[roles.role==label].sort_values('availability_rank');assert np.array_equal(ranks,records.availability_rank) and np.array_equal(truth,records.y)
    crows=roles[roles.role=='C'].sort_values('availability_rank');assert np.array_equal(p,crows.raw_p.to_numpy()) or np.allclose(p,crows.raw_p.to_numpy(),rtol=0,atol=1e-16)
    assert np.all(rank<100000) and np.all(rr<100000) and neighbors.shape==(200,15) and np.isin(neighbors,rr).all()
    h0=(p>=.5).astype(int);q=np.where(h0==1,ls,1-ls);s=np.maximum(p,1-p);prior=float(ry.mean());assert prior==23/200
    assert ((q>=0)&(q<=1)).all();hl=(q>=.5).astype(int);strong=np.maximum(q,1-q)>=.75
    h2=np.where((hl!=h0)&strong,hl,h0);hp=int(prior>=.5);h3=np.where((hp!=h0)&(max(prior,1-prior)>=.75),hp,h0)
    labels={'E0':h0,'E1':hl,'E2':h2,'E3':h3}
    sample=pd.DataFrame(dict(availability_rank=rank,y=y,raw_p=p,raw_s=s,original_local_support=ls,q_A=q,q_prior=prior,**labels))
    conf=[dict(family=m,**counts(y,h)) for m,h in labels.items()];harm=[]
    for m,h in labels.items():
        if m=='E0':continue
        for src,dst in [(0,1),(1,0)]:
            changed=(h0==src)&(h==dst);wrongright=int((changed&(h0!=y)&(h==y)).sum());rightwrong=int((changed&(h0==y)&(h!=y)).sum())
            harm.append(dict(family=m,direction='B_to_A' if src==0 else 'A_to_B',changed=int(changed.sum()),wrong_to_right=wrongright,right_to_wrong=rightwrong));assert wrongright+rightwrong==changed.sum()
    selected={};allc=[];feas=[]
    for m,fam,qgate in [('E0-local','E0',q),('E1-local','E1',q),('E2-local','E2',q),('E3-local','E3',q),('E3-prior','E3',np.full(200,prior))]:
        rule,rows,action=select(m,labels[fam],y,s,qgate,grid);selected[m]=rule;allc.extend(rows);sample['action_'+m]=action
        feas.append(dict(method=m,status=rule['status'],feasible_B_candidates=rule['feasible_enabled_candidates']['B'],feasible_A_candidates=rule['feasible_enabled_candidates']['A'],**rule['metrics']))
    e0=selected['E0-local'];assert e0['id']==oldrule['id']
    for c in ['B','A']:
        for key in ['direction','threshold','u','accepted','errors','correct','risk']:assert e0[c][key]==oldrule[c][key]
    original=np.full(200,-1,dtype=int)
    for i,c in enumerate(['B','A']):original[(h0==i)&accept(oldrule[c],s,ls)]=i
    assert np.array_equal(original,sample['action_E0-local'].to_numpy())
    roles.to_csv(run/'calibration/sample_roles.csv',index=False)
    sample.to_csv(run/'calibration/C_sample_labels.csv',index=False)
    for name,rows in [('calibration_confusion',conf),('correction_harm',harm),('gate_candidates',allc),('gate_feasibility',feas)]:pd.DataFrame(rows).to_csv(run/f'calibration/{name}.csv',index=False,na_rep='NA')
    write(run/'calibration/selected_rules.json',dict(completed_at=now(),protocol='config/protocol.json',E0_reproduces_original_actions_and_counts=True,q_prior=prior,grid=grid,methods=selected))
    fmt=lambda v:'NA' if v is None else f'{100*v:.4f}%'
    lines=['# ToN local labels correction: calibration feasibility', '', '200R/200C with only a replicator of the original ToN with the existing 400 clause labels. The basic RF freeze allows the decision-making layer to change the final labels. C point-by-point support has been preserved, this time without recalculating the neighbourhood; The following labels rating has not been read.', '', f'R：177B/23A，q_prior={prior}; C:166B/34A. The original 12 point value grid, the distance from k=15 remains unchanged. The E0 receives the / abstention movement at each point; the selected ID and branch counts both reproduce the original G2.', '', '## Full acceptance of confusion and correction of damage', '', '| Decision-making family |TN|FN|FP|TP|', '|---|---:|---:|---:|---:|']
    for row in conf:lines.append('|'+row['family']+'|'+'|'.join(str(row[k]) for k in ['TN','FN','FP','TP'])+'|')
    lines+=['','| Decision-making family | Change direction | Change total number of |wrong → right|right → wrong|','|---|---|---:|---:|---:|']
    for row in harm:lines.append(f"|{row['family']}|{row['direction']}|{row['changed']}|{row['wrong_to_right']}|{row['right_to_wrong']}|")
    lines+=['','## Door Control Candidate and Selection Rules','','The s in the gate control is always the max(p,1-p) of the original RF rather than the confidence probability of the modified labels. E0/E1/E2/E3-local uses real local support for the final candidate labels; E3-prior is supported by constant preconditions. Each activated branch requires acceptance of ≥10 and operating conditions risk ≤5%.','','| Method | Feasible B candidates | Feasible A candidates | Selected state | B accepted/errors/RB | A accepted/errors/RA |','|---|---:|---:|---|---|---|']
    for m,z in selected.items():
        b=z['B'];a=z['A'];lines.append(f"|{m}|{z['feasible_enabled_candidates']['B']}|{z['feasible_enabled_candidates']['A']}|{z['status']}|{b['accepted']}/{b['errors']}/{fmt(b['risk'])}|{a['accepted']}/{a['errors']}/{fmt(a['risk'])}|")
    lines+=['','The candidate number is calculated by the × threshold index ×u, excluding OFF and does not combine to produce the same action. the final statistics of the action of the risk subset as corrected; OFF risk is NA.','']
    for m,z in selected.items():lines.append(f"- {m}：B={z['B']['id']}（t={z['B']['threshold']}, u={z['B']['u']}）；A={z['A']['id']}（t={z['A']['threshold']}, u={z['A']['u']}）。")
    lines+=['','## Possible Conclusions and Stop Borders','']
    local_tp={row['family']:row['TP'] for row in conf if row['family'] in ['E1','E2']}
    if all(v==0 for v in local_tp.values()):conclusion='E1/E2 does not have the correct attack action; The current fixed neighbourhood does not provide the attack-correcting capabilities seen on C. Stop expanding the fixed neighbourhood labels correction and do not run pilot.'
    elif all(selected[m]['feasible_enabled_candidates']['A']==0 for m in ['E1-local','E2-local']):conclusion='The correct attack corrections occurred, but neither E1/E2 reached the attack candidate for accepting ≥10 and RA≤5%; There is no such thing as a two-pronged strategy. Preserving the correction and injury counts; The budget or declaration of change must be decided by way of alternative, this time stopped.'
    elif selected['E1-local']['status']=='TWO_ACTION_POLICY' and selected['E3-prior']['feasible_enabled_candidates']['A']==0:conclusion='E1-local establishes two branches of viable candidates on C: B accepts the 52 clause, incorrect 2 clause ((RB=3.8462%)); A accepts the 13 clause, the wrong 0 clause, the RA=0%. Its newly added 13 line correct attack correction does not appear in the E3 prefix, and E3-prior does not have a possible branch enabled. According to the results of the annex, E1 should be considered as an alternative to labels modification pilot validation, but only as an alternative to RF probability modification of baseline and freeze rules; It’s not being executed this time. E2 has not established dual branches: it fully accepts labels as equivalent to E3, E2-local as equivalent to E3-local, and only B branches are available. The above is only valid for repeated C calibration and cannot be claimed as validated or reliably repaired by independent.'
    else:conclusion='attacking the feasibility of candidates and the prior controversy; This round is only for the development results of C reuse, whether the expansion is subject to alternative authorization.'
    lines += [conclusion,'','The new method of replicating C is chosen for further development, not for independent verification or statistical risk assurance. The annex does not define the numerical threshold for the approximate effect threshold, so only the actual calculation of E3-prior is reported and the threshold is set after the accident. Subsequently, if the labels correction gains are compared, an alternative provision is required to allow only the RF probability of changing classification threshold to baseline; This round is not included.','', '## files', '']
    for name in ['config/protocol.json','config/06_Pilot_Interpretation_and_Next_Experiment_ZH.md','calibration/C_sample_labels.csv','calibration/calibration_confusion.csv','calibration/correction_harm.csv','calibration/gate_feasibility.csv','calibration/gate_candidates.csv','calibration/selected_rules.json','logs/commands.sh','src/run_calibration.py']:lines.append(f'- [{name}]({run/name})')
    lines+=['',f' The original input is from [ only.{old}/calibration]({old}/calibration) and the original freeze rules; The original run is unchanged. BoT, S2, S3, feedback or other models are not running.']
    (run/'reports/summary_ZH.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(conf).to_string(index=False));print(pd.DataFrame(harm).to_string(index=False));print(pd.DataFrame(feas).to_string(index=False));print(conclusion)

if __name__=='__main__':main()
