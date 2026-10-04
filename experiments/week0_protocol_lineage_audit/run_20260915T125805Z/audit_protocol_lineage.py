#!/usr/bin/env python3
"""Read-only lineage audit; never imports or executes historical experiment code."""
import csv, json, hashlib, sys, platform, subprocess, struct
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter,defaultdict
import numpy as np
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run'); OUT=Path(__file__).resolve().parent
BASE=ROOT/'outputs/post_p7_diagnostics'; PILOT=BASE/'week0_pilot/run_20260915T073452Z'; PRE=BASE/'week0_preflight/run_20260915T070731Z'
ROAD=ROOT/'docs/Research_Roadmap_Phase2_Conformal_Coordination.md'
now=lambda:datetime.now(timezone.utc).isoformat()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def table(name,data,fields=None):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(data[0]));w.writeheader();w.writerows(data)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def inventory():
 result={}
 for base in ['docs','configs','src','logs','datasets','outputs']:
  for p in (ROOT/base).rglob('*'):
   if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts:
    st=p.stat();result[str(p)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
 return result
print('START',now(),flush=True); before=inventory();dump('upstream_inventory_before.json',before)
candidates=[]
for p in (BASE/'week0_audit').glob('*/inference_validation.json'):
 j=read(p);s=j.get('summary',[])
 if sum(x.get('sampled_rows',0) for x in s)==53813 and all(x.get('bit_mismatch_count')==0 and x.get('q_set_change_count')==0 for x in s):
  v=read(p.parent/'audit_validation.json'); candidates.append((v.get('completed_at_utc',''),p.parent))
assert candidates
AUDIT=sorted(candidates)[-1][1]
dump('audit_identification.json',{'criterion':'53813 actual inference rows; bit_mismatch_count=0; q_set_change_count=0; completion time only after content qualification','candidates':[{'completed_at':t,'path':str(p)} for t,p in candidates],'selected':str(AUDIT)})
# Hash all already registered dependencies once; include actual code, reports and manifests.
registered={}; expected=[]
for folder in [PRE,PILOT,AUDIT]:
 for x in rows(folder/'source_hashes.csv'):
  p=Path(x['path']);registered[p]=x.get('version','SHA256 identifies content');expected.append((str(folder/'source_hashes.csv'),p,x['sha256']))
for folder in [PRE,PILOT,AUDIT]:
 for p in folder.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:registered.setdefault(p,'exact SHA256 snapshot')
for base in ['docs','configs','src/nids','outputs/round1_v2/config_snapshot','outputs/source_v2/P2','outputs/source_v2/preprocessing','outputs/round1_data_prep_v1','logs']:
 for p in (ROOT/base).rglob('*'):
  if p.is_file() and p.suffix in {'.md','.py','.json','.log','.csv','.sh'} and p.stat().st_size<4_000_000:registered.setdefault(p,'exact SHA256 snapshot')
stages={
 'A':BASE/'stage_A/run_20260914_A1','B1':BASE/'stage_B1/run_20260914_B1_01','B2':BASE/'stage_B2/run_20260914_B2_01',
 'Strict-K preflight':BASE/'strict_k_reply4/run_20260914_Cpreflight_01/preflight',
 'C1':BASE/'strict_k_reply4/run_20260915_C1_01','C2':BASE/'strict_k_reply4/run_20260915_C2_01'}
for folder in stages.values():
 for p in folder.rglob('*'):
  if p.is_file() and p.suffix in {'.md','.py','.json','.log','.sh'} and p.stat().st_size<10_000_000:registered.setdefault(p,'exact SHA256 snapshot')
hashes={};hashrows=[];missing=[];text_corpus={}
for i,(p,version) in enumerate(sorted(registered.items())):
 if not p.exists():missing.append(str(p));continue
 st=p.stat();digest=sha(p);hashes[p]=digest
 if p.suffix in {'.md','.py','.json','.log','.sh'} and st.st_size<10_000_000:
  text_corpus[p]=p.read_text(errors='replace')
  if p.suffix=='.json':
   try:version=str(json.loads(text_corpus[p]).get('version',version))
   except (ValueError,AttributeError):pass
 hashrows.append({'path':str(p),'version':version,'mtime_utc':datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat(),'mtime_ns':st.st_mtime_ns,'bytes':st.st_size,'sha256':digest,'hash_basis':'current bytes streamed; not timestamp inference'})
 if i%75==0:print('HASH',i,len(registered),str(p),flush=True)
table('source_hashes.csv',hashrows)
comparisons=[{'record':record,'path':str(p),'expected_sha256':e,'actual_sha256':hashes.get(p,'MISSING'),'status':'MATCH' if hashes.get(p)==e else 'MISMATCH_OR_MISSING'} for record,p,e in expected]
table('artifact_differences.csv',comparisons)
critical_mismatch=[x for x in comparisons if x['status']!='MATCH']
roadcompare=[x for x in comparisons if x['path']==str(ROAD)];dump('roadmap_version_comparison.json',{'current_sha256':hashes[ROAD],'comparisons':roadcompare,'same_execution_text':all(x['status']=='MATCH' for x in roadcompare),'professor_approval_of_exact_version':'NOT_VERIFIED; document status is not an independent approval record'})
# Retain exact evidence locations, rather than treating later summaries as original logs.
patterns=['argsort','kind="stable"','kind=\'stable\'','original_row','fit(','predict_proba','raw_probability','float32','float64','S_train','S_prob_cal','fill_values','131072','np.save','pickle.dump','OUT =','OUT=','external_history','Disclosure Statement']
extract=[]
for p,t in text_corpus.items():
 if p.suffix=='.py' or p.name in {'execution_v2.json','protocol_decisions.md','Professor_Progress_Report_After_P7.md'}:
  for ln,line in enumerate(t.splitlines(),1):
   if any(k in line for k in patterns):extract.append({'path':str(p),'line':ln,'text':line[:1500],'evidence_strength':'Static code' if p.suffix=='.py' else 'Direct verification of the original product (configuration)' if p.suffix=='.json' else 'Subsequent reports'})
table('code_evidence.csv',extract)
# Re-check the saved sampled evidence without repeating inference.
sample=rows(AUDIT/'frozen_sample_manifest.csv'); infer=rows(AUDIT/'inference_comparison.csv');feat=rows(AUDIT/'feature_comparison.csv'); sf=read(AUDIT/'sample_freeze.json');iv=read(AUDIT/'inference_validation.json')
keys=lambda rs:{(r['domain'],int(r['rank']),int(r['original_row'])) for r in rs}
roles=Counter();bydomain=Counter();timecount=Counter();batch=defaultdict(set);window=Counter();equal_ranks=defaultdict(set)
for r in sample:
 d=r['domain'];rank=int(r['rank']);bydomain[d]+=1;batch[d].add(rank//131072)
 for role in r['sample_roles'].split(';'):roles[role]+=1
 if d=='ToN':timecount[str(rank//6880000)]+=1
 if 'window17_equal_rank' in r['sample_roles']:
  window[(d,rank//10000)]+=1;equal_ranks[(d,rank//10000)].add(rank)
expected_windows=[('ToN',w) for w in range(2752)]+[('UNSW',w) for w in range(213,236)]
window_bad=[]
for d,w in expected_windows:
 wanted=set(np.linspace(w*10000,(w+1)*10000-1,17,dtype=np.int64).tolist())
 if not wanted.issubset(equal_ranks[(d,w)]):window_bad.append([d,w])
bit_errors=sum(struct.pack('>d',float.fromhex(r['cached_attack_hex']))!=struct.pack('>d',float.fromhex(r['recomputed_attack_hex'])) for r in infer)
sets_changed=sum(r['old_set']!=r['new_set'] for r in infer)
accepted=[r for r in infer if r['domain']=='ToN' and r['old_set'] in {'1','2'}]
feature_bad=sum(int(r['bit_mismatches'])+int(r['filled_input_bit_mismatches'])+int(r['value_or_missing_mismatches']) for r in feat)
check={'rows':len(sample),'unique_identity_rows':len(keys(sample)),'manifest_matches_frozen_sha256':hashes[AUDIT/'frozen_sample_manifest.csv']==sf['manifest_sha256'],'sample_and_inference_identity_equal':keys(sample)==keys(infer),'by_domain':dict(bydomain),'role_counts':dict(roles),'ToN_four_segment_sample_counts':dict(timecount),'producer_batch_indices':{d:sorted(s) for d,s in batch.items()},'complete_window_equal_rank_failures':window_bad,'full_evaluation_windows_checked':len(expected_windows),'ToN_accepted_in_sample':len(accepted),'probability_bit_mismatches_independently_read_from_hex':bit_errors,'set_mismatches':sets_changed,'feature_mismatches_in_saved_comparison':feature_bad,'freeze_before_inference':sf['frozen_at_utc']<iv['inference_started_at_utc'],'full_cache_reinference':False,'evidence':'saved sample/inference rows rejoined by identity; hex values bit compared independently; feature element proof uses prior CSV reconstruction and aggregate comparison','prior_audit_path':str(AUDIT)}
assert len(sample)==53813 and len(keys(sample))==53813 and not window_bad and len(accepted)==731 and bit_errors==sets_changed==feature_bad==0 and check['sample_and_inference_identity_equal'] and check['manifest_matches_frozen_sha256'] and check['freeze_before_inference']
dump('sample_coverage_verification.json',check)
print('SAMPLE EVIDENCE VERIFIED',flush=True)
# Exact roadmap citations and code anchors.
rt=ROAD.read_text();lines=rt.splitlines()
def loc(p,needle):
 t=Path(p).read_text(errors='replace').splitlines()
 for n,l in enumerate(t,1):
  if needle.lower() in l.lower():return f'{p}:{n}'
 return f'{p} (see full file; anchor not found: {needle})'
def excerpt(needle,following=0):
 for i,l in enumerate(lines):
  if needle.lower() in l.lower():return f'{ROAD}:{i+1}', '\n'.join(lines[i:i+1+following])
 raise ValueError(needle)
matrix=[]
def m(id,needle,cls,protocol,code,evidence,impact,advice,authority='roadmap',following=0,strength='Direct verification of the static code / of the raw product; The execution results are verified by independent audit'):
 l,q=excerpt(needle,following);matrix.append(dict(requirement_id=id,original_location=l,original_excerpt=q,protocol_location=protocol,code_location=code,execution_evidence=evidence,classification=cls,authority=authority,evidence_strength=strength,impact=impact,recommendation=advice))
p=str(PILOT/'protocol.json');ev=str(PILOT/'evaluate.py');im=str(PILOT/'implementation.py');va=str(AUDIT/'audit_validation.json')
m('W00','Before any full implementation','EXPLICIT_MATCH',p,ev,str(PILOT/'stage_Week0_pilot_report.md'),'only run RF + ToN Week0; Not expanded methods','Reservation scope')
m('W01','Load the existing all-49 RF','EXPLICIT_MATCH',p,ev,str(AUDIT/'model_reload.json'),'Fixed rf_All49 SHA256 corresponds to 49 column, classes=[0,1]','No exchange of assets')
m('W02','Run prediction sets on NF-ToN','EXPLICIT_MATCH',p,ev,str(PILOT/'evaluation_scope.csv') if (PILOT/'evaluation_scope.csv').exists() else str(PRE/'evaluation_scope.csv'),'Full 2752 window; Compatible with ToN requirements','Keep the original window')
m('W03','Compute split conformal calibration on S_prob_cal','EXPLICIT_MATCH',p,str(PILOT/'calibrate.py'),str(PILOT/'q_freeze.json'),'It is the source of calibration, Variant, A, Strict-0; Not using 400 target labels','Keep the independent q freeze')
m('W04',"detector's softmax probability",'AMBIGUOUS',p,im,str(AUDIT/'inference_validation.json'),'RF without softmax layer; Directly using RF[:,1] as directed by the user, not additional softmax;[:,0] and 1-p differ in float32.','In subsequent documentation, clarify the compatibility of the RF native probability with the secondary complement; Don’t change the algorithm','roadmap phrasing + clear instructions for users',1)
m('W05','Platt scaling — Platt','EXPLICIT_MATCH',p,ev,str(AUDIT/'producer_code_review.md'),'Just read raw_probability; Platt history but not participating in Week0','Historical Platt fitting alone does not establish that calibration is unusable')
m('W06','| S_prob_cal |','IMPLEMENTATION_DETAIL',p,str(ROOT/'outputs/round1_data_prep_v1/prepare_data.py'),str(PRE/'source_partition_audit.csv'),'The original 15% and unchanged;[1420000,1780000) taken as a whole by the whole 10k group and weighted as a whole without proportion','Distinguishing the overview ratio of freeze to actual partition','roadmap + History freeze protocol + user instructions')
m('W07','| S_train |','IMPLEMENTATION_DETAIL',p,str(ROOT/'src/nids/p2.py'),str(ROOT/'outputs/source_v2/models/rf_All49/training_manifest.json'),'the [0,1420000); The current RF fit does not use S_prob_cal; Undocumented human selection remains unverifiable','Preserving historical gaps','roadmap + History of the freeze protocol')
m('W08','α(x, y) = 1','EXPLICIT_MATCH',p,im,str(AUDIT/'independent_q_verification.json'),'y0=p,y1=1-p; No additional changes','Keeping the direct complement path')
m('W09','⌈(n+1)(1−ε)⌉/n quantile','IMPLEMENTATION_DETAIL',p,im,str(PILOT/'q_freeze.json'),'Limited sample order of the original; This round specifies the integers k=324001, uninterruptible, hex, recovery, iso-grouping.','The default quantile cannot be implemented as the only source','roadmap + has reviewed numerical contracts')
m('W10','≤ q̂','EXPLICIT_MATCH',p,im,str(AUDIT/'numerical_validation.json'),'Contains an equal number; Border Fraction sampling is consistent with the RF value contract','Not randomly disassembled ties')
m('W11','0.10','EXPLICIT_MATCH',p,im,str(PILOT/'q_freeze.json'),'The main epsilon=1/10; Not selective risk 10%','Not scanning epsilon')
m('W12','Window size: 10,000','IMPLEMENTATION_DETAIL',p,ev,str(PRE/'evaluation_scope.csv'),'The order of the original window, source end 5424, ToN excluding end 260; ToN does not exclude the initiation of 100000; Original unlisted endpoints','Scale for freeze range and user-performing refinement','roadmap + P7/ User instructions')
m('W13','Grouped bootstrap over windows','EXPLICIT_MATCH',p,str(PILOT/'bootstrap.py'),str(AUDIT/'statistics_validation.json'),'ToN2000, 10 window block, 2 window at the end; Sharing matrix','future risk guaranteed',following=1)
m('W14','Grouped bootstrap over windows','IMPLEMENTATION_DETAIL',p,str(PILOT/'bootstrap.py'),str(PILOT/'bootstrap_manifest.json'),'PCG64 seeds 20260915, 276 whole block extraction, linear 95% percentage, NA refined with the same replication differential value; The source is only 3 but not CI.','No temporary shortening of the source block length; Non-claimed source target matching CI','Users clearly statistically instruct / historical block length')
m('W15','Report: singleton rate','EXPLICIT_MATCH',p,ev,str(PILOT/'prediction_set_metrics.csv'),'Four sets of mutually exclusive counts, all source targets report true labels coverage','Difference between acceptance and prediction_set_coverage')
m('W16','Coverage (fraction accepted)','AMBIGUOUS',p,ev,str(PILOT/'selective_metrics.csv'),'coverage refers to the acceptance rate for the other phases of roadmap; Week0 names the two concepts separately to avoid confusion','Unified name of subsequent documentation')
m('W17','meaningfully more multi-label','EXPLICIT_MATCH',str(PILOT/'gate_definition.md'),str(PILOT/'report.py'),str(PILOT/'gate_assessment.md'),'The goal of empty is more, but not enough; Refusing to accept the wrong direction of the / error rate','Joint test loss review, no new value threshold',following=2)
m('W18','99%+ singleton','IMPLEMENTATION_DETAIL',str(PILOT/'gate_definition.md'),str(PILOT/'report.py'),str(PILOT/'gate_assessment.md'),'99%+ as an example; Not the only no-go condition; Four states have been audited and refined','This is not an example of a 5%/10% or a C2 being converted into gate.','roadmap + clear instructions for users')
m('W19','report findings to professor','NOT_VERIFIED',p,str(PILOT/'report.py'),str(PILOT/'stage_Week0_pilot_report.md'),'reports have been produced; Current records do not prove that confirmation has been sent or received and users are prohibited from automatically sending','Waiting for user research decisions; It’s not over.','roadmap + users do not send instructions',strength='/ failure to verify external communications')
m('W20','Do not proceed to Week 1','EXPLICIT_MATCH',p,str(PILOT/'report.py'),va,'Current delivery and directory average Week0; No automated Week1 input execution proof','Keep the stop.')
m('W21','minimum per-class count','IMPLEMENTATION_DETAIL',p,im,str(PRE/'preflight_report.md'),'Each class of 200 belongs to a succession of Mondrian; ToN343/57, CIC400/0 and BoT41/359 are both insufficient, but do not interfere with Strict0.','You can’t have future in Week0.','roadmap follow-up phase scope')
m('W22','pilot (Week 0) must pass before','IMPLEMENTATION_DETAIL',p,str(PILOT/'report.py'),str(PILOT/'gate_assessment.md'),'Subsequent full pre-registration of manifest is not a prerequisite for Week0; Explore already unblinding declaration reserved','Unconfirmed non-starter follow-up','Applicability at roadmap stage',following=1)
table('roadmap_compliance_matrix.csv',matrix)
# Dependency graph: concrete source producer and manifest, assessment scoped by evidence.
def findpaths(pattern):return ';'.join(str(x) for x in ROOT.glob(pattern))
lineage=[]
def edge(id,inp,producer,out,manifest,status,scope,gap):
 lineage.append({'edge_id':id,'input_assets':inp,'producer_code':producer,'output_assets':out,'manifest_or_protocol':manifest,'hash_link_status':status,'evidence_strength':'Direct verification of the static code + of the raw product; Previous independent audit execution record','verified_scope':scope,'unverified_or_risk':gap,'latest_audit_evidence':str(AUDIT/'audit_validation.json')})
edge('L01',str(ROOT/'datasets/NF-UNSW-NB15-v3.csv')+';'+str(ROOT/'datasets/NF-ToN-IoT-v3.csv'),str(ROOT/'outputs/round1_data_prep_v1/prepare_data.py'),str(ROOT/'outputs/round1_data_prep_v1'),str(ROOT/'outputs/round1_data_prep_v1/fold_manifest.json'),'current hashes match registered prior hashes','The original CSV end-time to original_row total 29885684 with stable ordering and only substitution; labels / Category 28115424 Total quantity consistent','identity is the only one that is not dependent on duplicate content or shared entities; It’s not a full body weight.')
edge('L02',str(ROOT/'outputs/round1_data_prep_v1'),str(ROOT/'outputs/round1_data_prep_v1/prepare_data.py'),str(PRE/'source_partition_audit.csv'),str(ROOT/'configs/execution_v2.json'),'frozen ranks and identity audit matched','train0:1420000;cal1420000:1780000;competence1780000:2130000;test2130000:2365424; train/cal/test primitive identity is non-intersecting','Unproven independent statistics between different identity')
edge('L03',str(ROOT/'datasets')+';'+str(ROOT/'configs/feature_views.csv'),str(ROOT/'src/nids/data/feature_store.py'),findpaths('outputs/**/feature_store_manifest.json'),str(ROOT/'configs/feature_views.csv'),'recorded feature/order hashes matched','49 columns are named in order of /; float64 read unlimited transitions after NaN to float32; CSV independent reconstructed 53813 sequential elements consistent','Not reconstructing all non-sampled feature elements; Absolute time without Label/Attack/ID/ in the 49 column')
edge('L04','UNSW S_train feature slice [0,1420000)',str(ROOT/'src/nids/data/preprocessing.py'),str(ROOT/'outputs/source_v2/preprocessing'),str(ROOT/'outputs/source_v2/preprocessing/preprocessing_manifest.json'),'stats hashes matched','37 numerical value median +12categorical mode( sub-value tie) , 49 replenishment value source training re-matching; RF not standardize; Calculate cross-domain and fill / without unit','Only verify the code with existing products, not prove that unrecorded historical operations never occurred')
edge('L05','frozen S_train X/y and execution_v2 RF params',str(ROOT/'src/nids/p2.py')+';'+str(ROOT/'src/nids/models/base_models.py'),str(ROOT/'outputs/source_v2/models/rf_All49/model.pkl'),str(ROOT/'outputs/source_v2/models/rf_All49/training_manifest.json'),'model SHA matches preflight/pilot/audit and StrictK checkpoint','RF fit with S_train only; 300 trees/depth16/leaf2/sqrt/n_bins128/bootstrap/seed20260905; There is no RF eval_set or automatic selection cycle; cuML26.8.0','Unregistered person for choice unknown; independent training stdout not detected, cannot be replaced with post-report')
edge('L06','same model + feature store + frozen preprocessing',str(ROOT/'src/nids/replay/predictions.py'),str(ROOT/'outputs/prediction_cache_v2/{UNSW,ToN}/rf_All49.npy'),str(ROOT/'outputs/prediction_cache_v2/{UNSW,ToN}/rf_All49_manifest.json'),'model/cache/current snapshot hashes matched; producer-era full cryptographic chain incomplete','131072 in continuous batches, predict_proba[:,1]float32 in intact f64; Platt other field; The last batch is ensuring stay; 53813 Reasoning for gradual matching','Caching without embedded line ID, depending on freeze location mapping; Sampling is not a complete rethink')
edge('L07','UNSW raw cache [1420000,1780000) + independent labels',str(PILOT/'calibrate.py'),str(PILOT/'q_freeze.json'),str(PILOT/'protocol_freeze.json'),'exact q hex/score hash independently matched','360000 real source score, k324001 , q freeze earlier than evaluate; Historical Platt Not Called','the history of the targets known during exploration; Can’t name the new outer layer independent verification')
edge('L08','frozen q + raw caches + original labels/windows',ev,str(PILOT/'ToN_window_counts.npz')+';'+str(PILOT/'prediction_set_metrics.csv'),p,'all audited hashes and 444160 tensor cells /282006 CSV fields match','The source 235424/ToN27520000 refers to the total quantity of references; The old trace without ph, not to mention reading the old ph collected verification','A verified downstream cannot prove caching production alone; Added with L03/L06 sampling')
edge('L09','ToN original 2752 window counts',str(PILOT/'bootstrap.py'),str(PILOT/'bootstrap_multiplicity.npy'),str(PILOT/'bootstrap_manifest.json'),'all 2000 replicate checks reproduced in prior audit','276 end blocks 2 window, domain sharing, divider /NA/130000 field / difference is consistent','Fixed block length applicability, stability and future range guarantees not proven')
table('asset_lineage.csv',lineage)
# Historical dependencies and exposure, distinguish actual new artifacts from contamination.
history=[]
for stage,created,readwk,impact,limits in [
 ('A','the Raw/Platt indicator and the audit itself; Add two MLP sources of reasoning, diagnosis only','Stage A detector_raw_platt_metrics.csv is used for the same range of RF background counts, without generating a q/ probability','A numerical contamination has not been detected; The B class result is unblinding; C class MLP difference without RF numeric dependence','Static writing paths are at this stage; It is impossible to prove that the whole of history has never been covered by the current timeline.'),
 ('B1','entropy/u scores, curves and polynomials','Background report / hash, not in q or prediction set','A class not found; The B class is known to have been misidentified; C historical rating is not directly dependent on current','Its protected record is mainly size/mtime and is not equal to the entire historical content hash'),
 ('B2','18 All49 f(u) fit Assets at this stage models directory','not using its f(u) values; Just the background','New assets are not RF recycled; A class not found; B class known fit / selection results; C class not currently reading f(u)','2710 upstream file size/mtime protected, cannot be claimed as full SHA'),
 ('Strict-K preflight','400 Objectives calibration identity and range manifest','Just understand the historical boundaries, not read the target labels as this round of calibration.','A class not found; B class target labels is known; Insufficient 200/ class only affects future Mondrian','model SHA checks can be connected; Early checks on cache are not all about SHA'),
 ('C1','400 target labels fit of classification threshold with shrinkage range bootstrap','Not using the C1 threshold or its bootstrap matrix','A class not found; Known target outcomes of B class; C class 100000 start exclusion of inheritance','Uploaded size/mtime minus the SHA file; There is no historical large cache of hash coverage'),
 ('C2','independent acceptance threshold / risk evaluation, using B2 and the matrix of the shrinkage range','No C2 threshold, 5%/10%gate or matrix','A class not found; B class is already unblinding; C class does not currently rely on C2 acceptance settings','Writing in this stage; Not to be confused with Upstream hash Protection Extended Expression as a complete historical guarantee')]:
 folder=stages[stage];codes=sorted(str(x) for x in folder.glob('*.py'));vals=sorted(str(x) for x in folder.rglob('*validation*.json'))
 history.append({'stage':stage,'path':str(folder),'read_existing_assets':True,'created_assets':created,'overwrite_current_dependencies':'NOT_FOUND within static code, available checkpoints and current hashes; all-time overwrite history NOT_VERIFIED','Week0_actual_dependency':readwk,'configuration_information_use':impact,'evidence':';'.join(codes+vals),'evidence_strength':'The static code / original hash checkpoint / performs verification records; Post-report assistance','limits':limits})
table('historical_stage_impact.csv',history)
# Focused evidence-gap search, not blanket re-experimentation.
search_roots=['docs','configs','logs','outputs/outer_v2/P7']
proc=subprocess.run(['rg','-n','external_history|USER_DECLARATION| Exposure Statement Outside of the | Project',*search_roots,'--glob','*.md','--glob','*.json','--glob','*.log'],cwd=ROOT,text=True,capture_output=True)
(OUT/'external_history_search.log').write_text(proc.stdout+proc.stderr)
reply_files=sorted(str(p) for p in ROOT.rglob('*') if p.is_file() and 'reply4' in p.name.lower() and OUT not in p.parents)
summary=Path('/mnt/c/Users/ysy/Desktop/literature/StrictK_A_B1_B2_C1_C2_Integrated_Summary_20260915.md')
dump('material_search.json',{'Reply4_named_files':reply_files,'Reply4_original':'MISSING: located experiment plan explicitly disclaims being original Reply4','integrated_summary_expected_Windows':r'C:\Users\ysy\Desktop\literature\StrictK_A_B1_B2_C1_C2_Integrated_Summary_20260915.md','wsl_path':str(summary),'wsl_exists':summary.exists(),'summary_role':'background only; separate stage evidence retained','missing_registered_paths':missing})
unresolved=[]
def u(id,item,status,impact,evidence,action):unresolved.append({'id':id,'item':item,'status':status,'impact':impact,'evidence':evidence,'evidence_strength':'It is not possible to verify the existence of evidence.','recommended_action':action})
u('U01','Primary Reply4','MISSING','Failure to verify the original requirements for teaching; Do not deny the readable roadmap and the user’s explicit execution of the instructions',str(ROOT/'docs/StrictK_Reply4_Experiment_Plan.md'),'To supplement the original communication if required by the attribution professor; No running backs')
u('U02','Independent Windows A–C2 integrated summary','MISSING' if not summary.exists() else 'FOUND','Background document missing;original stage code/reports reviewed',str(summary),'Can be synchronized later;not a numeric blocker')
u('U03','Professor recognizes specific roadmap version /Week0 confirmation','NOT_VERIFIED','Approved direction notation is not an external validation certificate; Week1 still can’t start automatically',str(ROAD)+':6','Waiting for user research decisions and clear confirmation')
u('U04','Exposure statements from past P7 projects','MISSING','freeze configuration requirements for supplementary registration; Lack of record limiting historical selection of independent sex, not proving that there has been a deactivation of the model',str(ROOT/'configs/execution_v2.json')+':388;'+str(ROOT/'docs/Professor_Progress_Report_After_P7.md')+':148','Completing the original statement / record; Continued exploration of the wheel')
u('U05','RF selection and test history undocumented','NOT_VERIFIED','The current code / manifest does not show S_prob_cal driven RF selection, and cannot prove that no unrecorded attempts have ever been made.',str(ROOT/'src/nids/p2.py'),'If independent is chosen as a claim, the complementary model is chosen to register; Lessons not to give negative results')
u('U06','Full features reconstructed and full cache re-reasoned','NOT_VERIFIED','53813 sampling is not consistent with all 27.5M models.',str(AUDIT/'inference_validation.json'),'No exception and no expansion; future If the full quantity is guaranteed, the independent methodology is required')
u('U07','All major assets of the original production period hash and independent RF training stdout','NOT_VERIFIED','manifest/ configuration for / training entities and later checkpoint connections; It’s not a perfect historical chain.',str(ROOT/'outputs/source_v2/models/rf_All49/training_manifest.json'),'maintaining the strength of evidence; If you find the original log file, it is forbidden to add it.')
u('U08','Repeat content / Sharing entities and time dependence','NOT_VERIFIED','The original line identity separation has been confirmed; It’s not a physical or statistical assurance of independent, it’s not a confirmation of a leak.',str(AUDIT/'information_isolation_validation.json'),'No IID or future domain guarantee is claimed; There are no specific anomalies in the total weight of / to change partition.')
u('U09','Fixed 10 window blocks long enough and future detects probability','NOT_VERIFIED','bootstrap is conditional on the current model /q and sequence; TP range 0 does not prove future probability 0',str(AUDIT/'statistics_validation.json'),'Limiting interpretation, not adjusting block length to accommodate conclusion')
table('unresolved_items.csv',unresolved)
dump('expanded_checks.json',[{'trigger_evidence':'execution_v2 external_history_action=USER_DECLARATION_REQUIRED_BEFORE_P7 but no linked declaration','object':'docs/configs/logs/P7 textual records','question':'Is original external result-exposure declaration available?','stop_condition':'bounded corpus rg search completed; no experiments','result':'No standalone declaration found; historical professor report explicitly records same gap; no numerical rerun justified'}])
# Mechanism claims are simple consequences of already independently verified distributions.
qj=read(PILOT/'q_freeze.json');print('Q keys',list(qj),flush=True)
q=float.fromhex('0x1.51bb5a0000000p-22')
pr=rows(AUDIT/'probability_score_diagnostics.csv');ton=[r for r in pr if r['domain']=='ToN' and r['field']=='attack_probability'];pmax=max(float(r['max']) for r in ton)
dump('mechanism_verification.json',{'q':q,'q_hex':q.hex(),'one_minus_q_hex':(1-q).hex(),'ToN_max_attack_probability':pmax,'ToN_max_attack_probability_hex':pmax.hex(),'minimum_candidate_attack_score':1-pmax,'Attack_membership_impossible_on_observed_ToN':1-pmax>q,'evidence':str(AUDIT/'probability_score_diagnostics.csv'),'causality':'Observed marginal distribution, label imbalance and ties explain rank mechanically; causes of domain/model probability behavior remain hypotheses','future_attack_detection_probability':'NOT_IDENTIFIED'})
after=inventory();dump('upstream_inventory_after.json',after)
changes=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)]
# Rehash small critical protocol/model/code/report inputs at end. Large inputs are streamed once and stat checked thereafter.
end_mismatch=[];end_hash_count=0
for p,h in hashes.items():
 if p.stat().st_size<=10_000_000:
  end_hash_count+=1
  if sha(p)!=h:end_mismatch.append(str(p))
assert not critical_mismatch and not changes and not end_mismatch
validation={'version':'week0-protocol-lineage-audit-v1.0','completed_at_utc':now(),'overall_conclusion':'B. found no substantive problems, but there is a clear evidence gap','numerical_substantial_issue_found':False,'roadmap_current_matches_execution_hashes':all(r['status']=='MATCH' for r in roadcompare),'roadmap_matrix_classifications':dict(Counter(x['classification'] for x in matrix)),'frozen_hash_comparisons':len(comparisons),'frozen_hash_mismatches':critical_mismatch,'current_hashed_assets':len(hashrows),'read_text_sources':len(text_corpus),'sample_evidence_checks':check,'historical_approval_and_selection':'NOT_VERIFIED; see unresolved_items.csv','explicit_roadmap_deviations_found':sum(x['classification']=='DEVIATION' for x in matrix),'old_artifact_stat_changes':changes,'end_rehashed_small_asset_count':end_hash_count,'end_hash_changes':end_mismatch,'large_input_protection':'current full hash vs historical recorded hash; before/after metadata; not full end rehash','prior_NO_GO_EVIDENCE_supported':True,'scope':'rf_All49 / ToN / Variant A / Strict-0 / epsilon1/10 / frozen q only; exploratory previously unblinded data','new_inference_or_experiments':False,'Week1_started':False,'professor_message_sent':False,'isolated_rerun_required':False,'next_step':'stop; research review and original documentary evidence only'}
dump('audit_validation.json',validation)
dump('environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'audit_device':'CPU: hashes, text and saved evidence only; no inference','original_inference_environment_evidence':str(AUDIT/'model_reload.json'),'cwd':str(ROOT),'started_run_id':OUT.name,'completed_at':now()})
print('COMPLETE',now(),'sources',len(hashrows),'comparisons',len(comparisons),flush=True)
