"""Finalize provenance joins and delivery checks; no experiment execution."""
from pathlib import Path
import csv,json,hashlib,struct
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;R=O.parents[3];B=R/'outputs/post_p7_diagnostics';P=B/'week0_pilot/run_20260915T073452Z';A=B/'week0_audit/run_20260915T075634Z'
def j(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def table(p,x):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert R==Path('/home/karri/projects/NIDSProject_Professor_Run'),R
hashes={x['path']:x['sha256'] for x in rows(O/'source_hashes.csv')}
check=[]
for folder in [P,A]:
 for r in rows(folder/'deliverable_hashes.csv'):
  p=Path(r['path']);h=hashes.get(str(p)) or sha(p)
  check.append({'record':str(folder/'deliverable_hashes.csv'),'path':str(p),'expected_sha256':r['sha256'],'actual_sha256':h,'status':'MATCH' if h==r['sha256'] else 'MISMATCH'})
assert all(x['status']=='MATCH' for x in check)
table(O/'historical_deliverable_hash_comparisons.csv',check)
model=j(R/'outputs/source_v2/models/rf_All49/training_manifest.json');pre=j(R/'outputs/source_v2/preprocessing/preprocessing_manifest.json');store=j(R/'outputs/feature_store_v2/feature_store_manifest.json');q=j(P/'q_freeze.json');proto=j(P/'protocol.json')
links={'feature_order_equal_model_preprocessing_store':model['feature_names']==pre['feature_order']==store['feature_order'],'feature_count':len(model['feature_names']),'forbidden_identity_label_features':sorted(set(model['feature_names']) & {'Label','Attack','ID','original_row','FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS','IPV4_SRC_ADDR','IPV4_DST_ADDR'}),'train_and_preprocessing_range_equal':model['training_rank_start']==pre['fit_rank_start']==0 and model['training_rank_end_exclusive']==pre['fit_rank_end_exclusive']==1420000,'q_protocol_hash_match':q['protocol_sha256']==sha(P/'protocol.json'),'q_code_hash_checks':{name:sha(P/name)==h for name,h in q['code_sha256'].items()},'q_expected_hex':q['q_serialized']['value']=='0x1.51bb5a0000000p-22','q_exact_k':q['k_one_based']==((q['n']+1)*9+9)//10==324001,'cache_manifest_checks':{}}
for d,n in [('UNSW',2365424),('ToN',27520260)]:
 c=j(R/f'outputs/prediction_cache_v2/{d}/rf_All49_manifest.json');f=j(R/f'outputs/feature_store_v2/{d}_manifest.json')
 links['cache_manifest_checks'][d]={'model_id_match':c['model_id']==model['model_id'],'feature_names_match':c['feature_names']==model['feature_names'],'feature_indices_match':c['feature_indices']==model['feature_indices'],'row_count_match':c['rows']==f['rows']==n,'CSV_source_match':f['source_csv']==str(R/f'datasets/NF-{ "UNSW-NB15" if d=="UNSW" else "ToN-IoT"}-v3.csv')}
assert links['feature_order_equal_model_preprocessing_store'] and not links['forbidden_identity_label_features'] and links['train_and_preprocessing_range_equal'] and links['q_protocol_hash_match'] and all(links['q_code_hash_checks'].values()) and links['q_exact_k'] and links['q_expected_hex']
assert all(all(c.values()) for c in links['cache_manifest_checks'].values())
dump(O/'manifest_join_verification.json',links)
# Upgrade code locations from file-only to concrete first relevant line.
m=rows(O/'roadmap_compliance_matrix.csv')
anchors={'implementation.py':'def ', 'evaluate.py':'raw_probability','calibrate.py':'threshold','bootstrap.py':'PCG64','report.py':'NO_GO','prepare_data.py':'S_prob_cal','p2.py':'train_one'}
for x in m:
 p=Path(x['code_location'])
 if p.exists():
  ls=p.read_text().splitlines();needle=anchors.get(p.name,'def ');ln=next((i for i,l in enumerate(ls,1) if needle in l),None)
  if ln is None:ln=next((i for i,l in enumerate(ls,1) if l.startswith('def ')),1)
  x['code_location']=str(p)+':'+str(ln)
table(O/'roadmap_compliance_matrix.csv',m)
# Clarify DNS_QUERY_ID is a legitimate frozen protocol feature, not CSV row identity.
f=O/'write_report.py';txt=f.read_text().replace('Not mixed with Label, Attack, ID, IP or absolute ordering time','Unmixed Label, Attack, identity line ID, IP address or absolute sequencing time (the DNS_QUERY_ID in the original 49 column is already a protocol field, not CSV line ID)');f.write_text(txt)
v=j(O/'audit_validation.json');v['historical_deliverable_hash_checks']=len(check);v['historical_deliverable_hash_mismatches']=0;v['manifest_join_verification']='PASS; see manifest_join_verification.json';v['layers']={'roadmap_explicit_Week0':'PASS within stated implementation details and textual ambiguities','upstream_current_asset_links':'PASS current manifests, feature order and recorded hashes; production-era complete cryptographic history NOT_VERIFIED','prior_independent_audit_evidence':'PASS at full label/replay and sampled feature/inference scopes','historical_selection_and_approval':'NOT_VERIFIED','numerical_mechanism_and_gate':'PASS current configuration NO_GO supported','statistical_implementation':'PASS prior independently reproduced replicates; temporal adequacy NOT_VERIFIED'};dump(O/'audit_validation.json',v)
print('Historical deliverable hashes:',len(check),'all MATCH; explicit manifest joins PASS')
