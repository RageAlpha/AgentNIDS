#!/usr/bin/env python
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,argparse,shlex,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from fixed_core import *
from frozen_functions import gate,choose,tests,boot_metrics,GRID
from io_utils import savej,saven,savecsv
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run');BASE=ROOT/'outputs/post_p7_diagnostics'
T1=BASE/'ton_recency_learner_pilot_v1/run_20260929T120812Z_zIglWd';T2=BASE/'ton_second_refresh_pilot_v1/run_20260929T122552Z_d10acc';OLD=BASE/'reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr'
BOUNDS=[20680000,22390000,24100000,25810000,27520000];SEEDS=[20260930,20261001,20261002];OBJECTIVES=['ENC_BCE','ENC_FCE','ENC_FSC'];HISTORY=['SOURCE_RF_ALL','T1_LOCAL_ALL','T1_LOCAL_GATED','SOURCE_GATE_T1','ALWAYS_A','ALWAYS_B']
PAIRS=[(l,r,layer) for l,r in [('ENC_FCE','ENC_BCE'),('ENC_FSC','ENC_BCE'),('ENC_FSC','ENC_FCE')]+[(o,'T1_LOCAL') for o in OBJECTIVES] for layer in ['ALL','GATED']]+[(o,'SOURCE_RF_ALL','ALL') for o in OBJECTIVES]+[(o,'SOURCE_GATE_T1','GATED') for o in OBJECTIVES]

def modelname(o,s):return f'{o}_{s}'
def actionname(o,s,layer):return modelname(o,s)+'_'+layer if o in OBJECTIVES else o+'_'+layer if o=='T1_LOCAL' else o
class Encoder(nn.Module):
    def __init__(self,sizes,nfamilies):
        super().__init__();self.emb=nn.ModuleList([nn.Embedding(n,4) for n in sizes]);self.backbone=nn.Sequential(nn.Linear(81,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,32));self.binary=nn.Linear(32,2);self.family=nn.Linear(32,nfamilies)
    def forward(self,num,cat):
        x=torch.cat([num]+[e(cat[:,i]) for i,e in enumerate(self.emb)],1);h=self.backbone(x);z=h/h.norm(dim=1,keepdim=True).clamp_min(1e-12);return z,self.binary(z),self.family(z)
def supcon(z,f):
    n=len(z);eye=torch.eye(n,dtype=torch.bool,device=z.device);positive=f[:,None].eq(f[None,:])&~eye;cnt=positive.sum(1);valid=cnt>0
    if not valid.any():return z.new_zeros(()),cnt
    logits=z@z.T/.1;den=torch.logsumexp(logits.masked_fill(eye,-torch.inf),dim=1);pos=(logits*positive).sum(1)/cnt.clamp_min(1);return (den-pos)[valid].mean(),cnt

def source_features(ranks,p):
    x=np.asarray(np.load(p['source_files']['features'],mmap_mode='r')[ranks],np.float64);fill=np.load(p['source_files']['statistics'])['fill'];x=np.where(np.isfinite(x),x,fill);require(np.isfinite(x).all(),'source imputation nonfinite');return x

def encode_input(x,stats,vocabs,cat,num):
    n=np.clip((x[:,num]-stats['mean'])/stats['std'],-10,10).astype(np.float32);cs=[]
    for j,v in zip(cat,vocabs):
        values=x[:,j];pos=np.searchsorted(v,values);ok=(pos<len(v))&(v[np.minimum(pos,len(v)-1)]==values);cs.append(np.where(ok,pos+1,0))
    return n,np.column_stack(cs).astype(np.int64)

def preflight(run):
    p1=readj(T1/'config/protocol.json');rf=readj(ROOT/'outputs/source_v2/models/rf_All49/training_manifest.json');mlp=readj(ROOT/'outputs/source_v2/models/mlp_All49/training_manifest.json');fm=readj(ROOT/'outputs/feature_store_v2/feature_store_manifest.json');sm=readj(ROOT/'outputs/feature_store_v2/UNSW_manifest.json');pm=readj(ROOT/'outputs/source_v2/preprocessing/preprocessing_manifest.json')
    require(fm['row_order']=='availability_sorted_rank' and rf['training_partition']==mlp['training_partition']=='S_train','source rank/split semantics');require(rf['feature_names']==mlp['feature_names']==fm['feature_order']==p1['input_assets']['feature_order'],'feature alignment')
    splits=pd.read_csv(Path(fm['rank_source'])/'split_manifest.csv');ss=splits[(splits.domain=='UNSW')&(splits.scheme=='source')];tr=ss[ss.partition=='S_train'].iloc[0];require(int(tr.rank_start)==rf['training_rank_start']==mlp['training_rank_start'] and int(tr.rank_end_exclusive)==rf['training_rank_end_exclusive']==mlp['training_rank_end_exclusive'],'source training split')
    require(set(ss.partition)=={'S_train','S_prob_cal','S_competence','S_source_test'},'unexpected source roles: inspect before assigning validation')
    alltrain=np.arange(int(tr.rank_start),int(tr.rank_end_exclusive),dtype=np.int64);perm=np.random.Generator(np.random.PCG64(20261011)).permutation(len(alltrain));nv=len(alltrain)//10;vpool=np.sort(alltrain[perm[:nv]]);tpool=np.sort(alltrain[perm[nv:]])
    sample=lambda pool,seed,cap:np.sort(np.random.Generator(np.random.PCG64(seed)).choice(pool,min(cap,len(pool)),replace=False)) if len(pool)>cap else pool
    train=sample(tpool,20261012,50000);val=sample(vpool,20261013,10000);require(not np.isin(train,val).any(),'source overlap')
    original=np.load(Path(fm['rank_source'])/'UNSW_sorted_original_rows.npy',mmap_mode='r');require(len(np.unique(original[np.r_[train,val]]))==len(train)+len(val),'source stable original ids')
    roles=pd.read_csv(T1/'config/recent_sample_roles.csv').sort_values('availability_rank');tz=np.load(T1/'calibration/recent400.npz');require(np.array_equal(roles.availability_rank,tz['rank']),'T1 rank identity');ty=tz['y'];require(np.bincount(ty[roles.role=='R'],minlength=2).tolist()==[128,72] and np.bincount(ty[roles.role=='C'],minlength=2).tolist()==[123,77],'T1 support')
    cache=np.load(T2/'tables/evaluation_actions.npz');rank=cache['rank'];require(len(rank)==68400 and np.array_equal(rank,np.load(T2/'config/frozen_indices.npz')['rank']) and np.all(np.diff(rank)>0),'evaluation identity');require(np.bincount((rank-BOUNDS[0])//10000).tolist()==[100]*684 and not np.isin(rank,roles.availability_rank).any(),'evaluation windows/disjoint')
    codebook={int(k):v.strip().lower() for k,v in sm['attack_codebook'].items()};require(len(set(codebook.values()))==len(codebook) and 'benign' in codebook.values() and not any(v in ['', 'unknown','nan','none'] for v in codebook.values()),'family names ambiguous')
    files={k:str(ROOT/'outputs/feature_store_v2'/v) for k,v in sm['files'].items()};files['statistics']=str(ROOT/'outputs/source_v2/preprocessing'/pm['statistics_file'])
    for path in files.values():require(Path(path).is_file(),'missing source asset '+path)
    device='cuda:0' if torch.cuda.is_available() else 'cpu';runtime=dict(interpreter=sys.executable,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=device,device_name=torch.cuda.get_device_name(0) if device.startswith('cuda') else platform.processor(),cuda=torch.version.cuda,cpu_threads=4)
    p=dict(experiment='ton_representation_pilot_v1',created_at=now(),runtime=runtime,source_files=files,source_manifest=str(ROOT/'outputs/feature_store_v2/UNSW_manifest.json'),source_split_manifest=str(Path(fm['rank_source'])/'split_manifest.csv'),family_provenance='feature_store.py reads CSV Attack as string, _encode_attack then shared prepared.sorted_rows; Label is binary truth',original_source_csv=sm['source_csv'],source_training_range=[int(tr.rank_start),int(tr.rank_end_exclusive)],validation_policy='original manifests have no encoder validation; split10% from S_train, leave probability-calibration/competence/test untouched; inherited source fill from original S_train',sampling=dict(split_seed=20261011,train_seed=20261012,val_seed=20261013,train_rows=len(train),validation_rows=len(val),generator='PCG64',source_train_pool=len(tpool),source_validation_pool=len(vpool)),source_codebook=codebook,feature_order=fm['feature_order'],cat=p1['input_assets']['categorical_indices'],num=p1['input_assets']['numeric_indices'],features=p1['feature_indices'],input='original source fill; selected source train numeric mean/population std clip[-10,10], independent category vocab UNK0 embeddings4;81dim',architecture='81-128-ReLU-64-ReLU-32, normalized z eps1e-12; binary32to2 family32toF all models',seeds=SEEDS,objectives=OBJECTIVES,training=dict(optimizer='Adam',lr=.001,weight_decay=.0001,batch_size=256,epochs=20,shuffle=True,drop_last=False,dtype='float32',aux_weight=.1,tau=.1,checkpoint='last',paired_initialization=True,paired_batch_order=True,no_scheduler=True,no_early_stopping=True),supcon='same source family different records positive; denominator all other records; valid anchors only; zero if none; single view no augmentation',target_roles=str(T1/'config/recent_sample_roles.csv'),target_labels=str(T1/'calibration/recent400.npz'),target_asset_manifest=p1['input_assets'],target_R_counts=[128,72],target_C_counts=[123,77],gate_grid=p1['gate_grid'],gate=p1['gate'],tie_order=p1['tie_order'],knn='cosine 1-clip(z dot zR,-1,1); k15 uniform; stable reference rank tie',baseline_actions=str(T2/'tables/evaluation_actions.npz'),baselines=HISTORY,bounds=BOUNDS,expected_rows=68400,pairs=PAIRS,bootstrap=dict(seed=20261014,replicates=1000,block_windows=10,per_segment=True,short_final=True,shared_all_seeds=True,seed_mean_requires_all_valid=True,min_valid_fraction=.9,percentiles=[2.5,97.5]),budget=dict(new_target_labels=0,cumulative_target_labels=1200,target_per_model=400,source_train_labels=len(train),source_validation_labels=len(val),family_supervision=['ENC_FCE','ENC_FSC'],evaluation_existing_labels=68400),workpoint=dict(max_risk=.05,min_utility=.1,min_action_count=100,min_action_windows=10,stable='overall and 3/4 segments,per seed'),interpretation='development reused target labels and outcomes; custom family objective intervention not full hierarchical malware reproduction; coefficient .1 not strength-matched optimum',references=['https://arxiv.org/abs/2004.11362','https://www.usenix.org/conference/usenixsecurity23/presentation/chen-yizheng'],stop='one bounded evaluation,no tuning/new labels/further experiments')
    p=clean(p)
    if (run/'config/protocol.json').exists():p['created_at']=readj(run/'config/protocol.json')['created_at']
    savej(run/'config/protocol.json',p);savecsv(run/'config/target_roles.csv',roles);saven(run/'config/evaluation_indices.npz',rank=rank)
    for name,rr in [('train',train),('validation',val)]:savecsv(run/f'source/{name}_indices.csv',pd.DataFrame(dict(availability_rank=rr,original_row=original[rr])))
    if not (run/'config/index_freeze.json').exists():writej(run/'config/index_freeze.json',dict(time=now(),indices_only_source_sampling=True,target_labels_previously_observed=True))
    require((run/'config/index_freeze.json').exists() and (run/'config/evaluation_indices.npz').exists(),'evaluation indices must be frozen before labels')
    ystore=np.load(files['binary_label'],mmap_mode='r');fstore=np.load(files['attack_code'],mmap_mode='r');ys={};families={}
    for name,rr in [('train',train),('validation',val)]:
        yy=np.asarray(ystore[rr],np.int64);codes=np.asarray(fstore[rr]);require(np.isin(codes,list(codebook)).all() and np.isin(yy,[0,1]).all(),'source label code');ff=np.array([codebook[int(k)] for k in codes]);require(np.array_equal(yy==0,ff=='benign'),'binary/family inconsistency');ys[name]=yy;families[name]=ff
    vocab=sorted(set(families['train']));mapping={v:i for i,v in enumerate(vocab)};savej(run/'config/source_label_mapping.json',dict(original=sm['attack_codebook'],normalization='strip and lower only; no family merging',train_family_to_id=mapping,unknown_validation_family_id=-1,provenance=p['family_provenance']))
    rows=[]
    for name in ['train','validation']:
        for family in sorted(set(families[name])):rows.append(dict(role=name,family=family,N=int((families[name]==family).sum()),binary_label=0 if family=='benign' else 1,known_in_train=family in mapping))
    savecsv(run/'source/label_support.csv',rows)
    sp=run/'source/preprocessing.npz';vp=run/'source/vocabularies.json'
    if not sp.exists():
        tx=source_features(train,p);mean=tx[:,p['num']].mean(0);std=tx[:,p['num']].std(0);std[std==0]=1;vocabs=[np.unique(tx[:,j]) for j in p['cat']];saven(sp,mean=mean,std=std);savej(vp,{p['feature_order'][j]:v.tolist() for j,v in zip(p['cat'],vocabs)})
    else:tx=source_features(train,p)
    stats=np.load(sp);vd=readj(vp);vocabs=[np.array(vd[p['feature_order'][j]]) for j in p['cat']];processed={}
    for name,xx in [('train',tx),('validation',source_features(val,p))]:
        n,c=encode_input(xx,stats,vocabs,p['cat'],p['num']);ff=np.array([mapping.get(v,-1) for v in families[name]],np.int64);processed[name]=(n,c,ys[name],ff)
    a=Assets(ROOT,'ToN',readj(OLD/'config/protocol_v1.json'));xx=a.features(roles.availability_rank.to_numpy(np.int64));tn,tc=encode_input(xx,stats,vocabs,p['cat'],p['num']);saven(run/'calibration/target400.npz',rank=roles.availability_rank.to_numpy(np.int64),y=ty,num=tn,cat=tc,role=roles.role.to_numpy(dtype='U1'))
    return p,processed,[len(v)+1 for v in vocabs],vocab,stats,vocabs,a

@torch.no_grad()
def validation(model,data,device,objective,seed,epoch):
    model.eval();probs=[];preds=[];fp=[];loss=0;n,c,y,f=data
    for start in range(0,len(y),1024):
        end=min(start+1024,len(y));z,b,ff=model(torch.as_tensor(n[start:end],device=device),torch.as_tensor(c[start:end],device=device));loss+=F.cross_entropy(b,torch.as_tensor(y[start:end],device=device),reduction='sum').item();preds.extend(b.argmax(1).cpu().tolist());fp.extend(ff.argmax(1).cpu().tolist())
    pred=np.array(preds);known=f>=0;tn=int(((y==0)&(pred==0)).sum());tp=int(((y==1)&(pred==1)).sum());nb=int((y==0).sum());na=int((y==1).sum())
    return dict(objective=objective,seed=seed,epoch=epoch,N=len(y),N_B=nb,N_A=na,TN=tn,FN=na-tp,FP=nb-tn,TP=tp,binary_CE=loss/len(y),balanced_accuracy=(tn/nb+tp/na)/2 if nb and na else None,family_known=int(known.sum()),family_unknown=int((~known).sum()),family_accuracy=float((np.array(fp)[known]==f[known]).mean()) if objective=='ENC_FCE' and epoch==20 and known.any() else None)

def train_one(run,p,data,sizes,familynames,objective,seed,device):
    name=modelname(objective,seed);directory=run/'models'/name;directory.mkdir(exist_ok=True);config=dict(protocol=p,objective=objective,seed=seed,sizes=sizes,family_names=familynames);savej(directory/'config.json',config)
    model=Encoder(sizes,len(familynames)).to(device);shared=run/'models'/f'initial_{seed}.pt'
    if not shared.exists():
        torch.manual_seed(seed);initial=Encoder(sizes,len(familynames));torch.save(initial.state_dict(),shared)
    init=torch.load(shared,map_location=device,weights_only=True);model.load_state_dict(init)
    if not (directory/'initial.pt').exists():torch.save({k:v.cpu() for k,v in init.items()},directory/'initial.pt')
    if (directory/'complete.json').exists():
        require((directory/'final.pt').exists(),'missing completed model');model.load_state_dict(torch.load(directory/'final.pt',map_location=device,weights_only=True));print(name,'resume completed',flush=True);return model
    if (directory/'TECHNICAL_TRAINING_FAILURE.json').exists():raise RuntimeError('previous technical failure, not silently retried '+name)
    train=data['train'];dataset=TensorDataset(*[torch.as_tensor(x) for x in train]);generator=torch.Generator().manual_seed(seed);loader=DataLoader(dataset,batch_size=256,shuffle=True,drop_last=False,generator=generator,num_workers=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);training=[];valid=[validation(model,data['validation'],device,objective,seed,0)];familyrows=[]
    try:
        for epoch in range(1,21):
            model.train();tot=np.zeros(4);valid_n=0;total=0;zero_batches=0;family_n=np.zeros(len(familynames),np.int64);family_valid=family_n.copy();family_pos=family_n.copy()
            for step,(num,cat,y,f) in enumerate(loader):
                num,cat,y,f=[x.to(device) for x in [num,cat,y,f]];optimizer.zero_grad(set_to_none=True);z,b,fh=model(num,cat);binary=F.cross_entropy(b,y);fce=F.cross_entropy(fh,f)
                if objective=='ENC_FSC':sc,cnt=supcon(z,f)
                else:
                    with torch.no_grad():sc,cnt=supcon(z.detach(),f)
                loss=binary+(.1*fce if objective=='ENC_FCE' else .1*sc if objective=='ENC_FSC' else 0)
                require(bool(torch.isfinite(loss)) and bool(torch.isfinite(z).all()),'nonfinite loss/embedding');loss.backward();require(all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in model.parameters()),'nonfinite gradient');optimizer.step()
                require(all(bool(torch.isfinite(v).all()) for v in model.parameters()),'nonfinite parameter')
                size=len(y);tot+=np.array([loss.item(),binary.item(),fce.item(),sc.item()])*size;total+=size;valid_n+=int((cnt>0).sum());zero_batches+=int(not (cnt>0).any());fc=f.cpu().numpy();pc=cnt.cpu().numpy()
                family_n+=np.bincount(fc,minlength=len(familynames));family_valid+=np.bincount(fc[pc>0],minlength=len(familynames));family_pos+=np.bincount(fc,weights=pc,minlength=len(familynames)).astype(np.int64)
            training.append(dict(objective=objective,seed=seed,epoch=epoch,rows=total,steps=step+1,total_loss=tot[0]/total,binary_CE=tot[1]/total,family_CE_diagnostic=tot[2]/total,family_SupCon_diagnostic=tot[3]/total,valid_anchor_fraction=valid_n/total,zero_anchor_batches=zero_batches))
            for k,family in enumerate(familynames):familyrows.append(dict(objective=objective,seed=seed,epoch=epoch,family=family,anchors=int(family_n[k]),valid_anchors=int(family_valid[k]),positive_pairs=int(family_pos[k])))
            print(name,'epoch',epoch,'binary_CE',tot[1]/total,'valid_anchor',valid_n/total,flush=True)
        valid.append(validation(model,data['validation'],device,objective,seed,20));torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},directory/'final.pt');savecsv(directory/'training.csv',training);savecsv(directory/'validation.csv',valid);savecsv(directory/'positive_support.csv',familyrows);writej(directory/'complete.json',dict(time=now(),status='TRAINING_COMPLETE',epochs=20,steps=20*len(loader),rows=len(train[2]),finite=True,checkpoint='epoch20'))
    except Exception as e:
        writej(directory/'TECHNICAL_TRAINING_FAILURE.json',dict(time=now(),objective=objective,seed=seed,epoch=epoch,step=step,error=repr(e),status='TECHNICAL_TRAINING_FAILURE'));torch.save(model.state_dict(),directory/'failed_state.pt');raise
    return model

@torch.no_grad()
def embed(model,n,c,device):
    model.eval();out=[]
    for start in range(0,len(n),2048):out.append(model(torch.as_tensor(n[start:start+2048],device=device),torch.as_tensor(c[start:start+2048],device=device))[0].cpu().numpy())
    z=np.concatenate(out);require(np.isfinite(z).all(),'nonfinite inference');return z

def neighbors(q,r,ry):
    d=1-np.clip(q@r.T,-1,1);nn=np.argsort(d,axis=1,kind='stable')[:,:15];return ry[nn].mean(1),nn

def synthetic_checks():
    checks=tests();z=torch.tensor([[1.,0],[0.,1],[1.,0]],requires_grad=True);f=torch.tensor([0,1,0]);loss,cnt=supcon(z,f);expected=torch.logsumexp(torch.tensor([0.,10.]),0)-10;require(torch.allclose(loss,expected,atol=1e-6) and cnt.tolist()==[1,0,1],'SupCon single-view definition');loss.backward();require(torch.isfinite(z.grad).all(),'SupCon gradient');zero,cnt=supcon(z,torch.arange(3));require(zero.item()==0 and cnt.sum()==0,'SupCon no positives');_,nn=neighbors(np.zeros((1,32),np.float32),np.zeros((20,32),np.float32),np.arange(20)%2);require(np.array_equal(nn[0],np.arange(15)),'cosine ties');v=metrics(window_counts(np.tile([0,1],50),np.zeros(100),np.full(100,-1),1)[0]);require(v['R_B'] is None and v['abstain_B']==v['abstain_A']==50,'OFF denominators');checks.update(SupCon_self_exclusion=True,SupCon_no_positive_zero=True,cosine_rank_ties=True,OFF_conservation=True);return checks

def prepare(run):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    p,data,sizes,families,stats,vocabs,a=preflight(run);device=torch.device(p['runtime']['device']);savej(run/'config/synthetic_checks.json',synthetic_checks())
    if (run/'tables/actions_complete.json').exists():
        require(readj(run/'calibration/selected_rules.json')['protocol']==p,'completed protocol');print('prepare complete; verified indices/config and skipped all training');return
    target=np.load(run/'calibration/target400.npz');r=target['role']=='R';c=~r;ty=target['y'];rank=np.load(run/'config/evaluation_indices.npz')['rank'];cache=np.load(p['baseline_actions']);actions=dict(rank=rank,segment=cache['segment'],window=cache['window'])
    for name in HISTORY:actions['action_'+name]=cache['action_'+name]
    rules={};candidates=[];support=[]
    for seed in SEEDS:
        # Store the exact source sampling sequence independent of model/loss RNG.
        orderpath=run/'source'/f'batch_order_{seed}.npz'
        if not orderpath.exists():
            loader=DataLoader(torch.arange(len(data['train'][2])),batch_size=256,shuffle=True,drop_last=False,generator=torch.Generator().manual_seed(seed));orders=np.array([torch.cat(list(loader)).numpy() for _ in range(20)]);saven(orderpath,indices=orders)
        for objective in OBJECTIVES:
            name=modelname(objective,seed);model=train_one(run,p,data,sizes,families,objective,seed,device);zp=run/'calibration'/f'{name}_target.npz'
            if zp.exists():zz=np.load(zp);zt=zz['z'];q=zz['C_score']
            else:
                zt=embed(model,target['num'],target['cat'],device);q,nn=neighbors(zt[c],zt[r],ty[r]);saven(zp,z=zt,C_score=q,C_neighbors=nn,rank=target['rank'])
            rulepath=run/'calibration'/f'{name}_rule.json'
            if rulepath.exists():saved=readj(rulepath);rule=saved['rule'];rows=saved['candidates']
            else:rule,rows=choose(q,ty[c],name);writej(rulepath,dict(time=now(),rule=rule,candidates=rows))
            rules[name]=rule;candidates+=rows
            for ro,truth,pr in [('R',ty[r],None),('C',ty[c],q>=.5)]:support.append(dict(method=name,objective=objective,seed=seed,role=ro,true_B=int((truth==0).sum()),true_A=int((truth==1).sum()),candidate_B=None if pr is None else int((~pr).sum()),candidate_A=None if pr is None else int(pr.sum()),B_threshold=rule['B']['threshold'] if ro=='C' else None,B_accepted=rule['B']['accepted'] if ro=='C' else None,B_errors=rule['B']['errors'] if ro=='C' else None,A_threshold=rule['A']['threshold'] if ro=='C' else None,A_accepted=rule['A']['accepted'] if ro=='C' else None,A_errors=rule['A']['errors'] if ro=='C' else None,status=rule['status'] if ro=='C' else 'R not self-evaluated'))
            actionpath=run/'tables'/f'{name}_actions.npz'
            if actionpath.exists():aa=np.load(actionpath);require(np.array_equal(aa['rank'],rank),'saved action ranks');score=aa['score']
            else:
                score=np.empty(len(rank))
                for start in range(0,len(rank),3000):
                    end=min(start+3000,len(rank));x=a.features(rank[start:end]);n,cat=encode_input(x,stats,vocabs,p['cat'],p['num']);z=embed(model,n,cat,device);score[start:end],_=neighbors(z,zt[r],ty[r])
                saven(actionpath,rank=rank,score=score,ALL=(score>=.5).astype(np.int8),GATED=gate(score,rule))
            actions['score_'+name]=score;actions['action_'+name+'_ALL']=(score>=.5).astype(np.int8);actions['action_'+name+'_GATED']=gate(score,rule)
            del model
    for filename,sourcefile in [('source_training','training.csv'),('source_validation','validation.csv'),('source_positive_support','positive_support.csv')]:savecsv(run/f'tables/{filename}.csv',pd.concat([pd.read_csv(run/'models'/modelname(o,s)/sourcefile,float_precision='round_trip') for s in SEEDS for o in OBJECTIVES],ignore_index=True))
    savecsv(run/'calibration/support.csv',support);savecsv(run/'calibration/candidates.csv',candidates);savej(run/'calibration/selected_rules.json',dict(protocol=p,methods=rules));saven(run/'tables/evaluation_actions.npz',**actions);writej(run/'tables/actions_complete.json',dict(time=now(),rows=68400,trained_models=9,all_actions_before_evaluation_truth=True));print('PREPARE COMPLETE',flush=True)

def evaluate(run):
    p=readj(run/'config/protocol.json');require((run/'tables/actions_complete.json').exists() and readj(run/'calibration/selected_rules.json')['protocol']==p,'prepare not frozen')
    if (run/'reports/decision.json').exists():print('evaluation complete; no overwrite');return
    a=np.load(run/'tables/evaluation_actions.npz');rank=a['rank'];require(np.array_equal(rank,np.load(run/'config/evaluation_indices.npz')['rank']) and len(rank)==68400,'evaluation identity')
    writej(run/'tables/evaluation_label_access.json',dict(time=now(),rows=68400,source='existing original pilot chunks',actions_complete=readj(run/'tables/actions_complete.json')['time']))
    yy=[];rr=[]
    for path in sorted((OLD/'pilot/ToN').glob('chunk_*.npz')):
        with np.load(path) as z:
            ranks=z['rank'];m=(ranks>=BOUNDS[0])&(ranks<BOUNDS[-1])
            if m.any():rr.append(ranks[m]);yy.append(z['y'][m])
    y=np.concatenate(yy);require(np.array_equal(np.concatenate(rr),rank),'truth alignment');seg=a['segment'];segw=seg.reshape(684,100)[:,0];counts={};overall={};segments=[]
    for key in a.files:
        if not key.startswith('action_'):continue
        name=key[7:];pred=a['action_SOURCE_RF_ALL'] if name=='SOURCE_GATE_T1' else a['action_'+name.replace('_GATED','_ALL')] if name.endswith('_GATED') else a[key]
        co=window_counts(y,pred,a[key],684);require(np.all(co[:,3]+co[:,5]+co[:,7]==co[:,1]) and np.all(co[:,4]+co[:,6]+co[:,8]==co[:,2]),'count conservation');counts[name]=co;overall[name]=dict(method=name,**metrics(co.sum(0)));overall[name]['coverage']=overall[name]['acceptance']
        for j in range(1,5):
            v=dict(method=name,segment=j,**metrics(co[segw==j].sum(0)));v['coverage']=v['acceptance'];segments.append(v)
    stable={n:v['target_status']=='JOINT_TARGET_MET' and sum(s['target_status']=='JOINT_TARGET_MET' for s in segments if s['method']==n)>=3 for n,v in overall.items()}
    blocks=[];strata=[]
    for j in range(1,5):
        idx=np.flatnonzero(segw==j);ids=[]
        for k in range(0,len(idx),10):ids.append(len(blocks));blocks.append(idx[k:k+10])
        strata.append(np.array(ids))
    rng=np.random.Generator(np.random.PCG64(20261014));draw=np.array([np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in strata]) for _ in range(1000)]);weight=np.array([np.bincount(d,minlength=len(blocks)) for d in draw]);saven(run/'tables/bootstrap_draws.npz',indices=draw);savecsv(run/'tables/bootstrap_blocks.csv',[dict(block=i,segment=int(segw[b[0]]),first_window=int(b[0]),windows=len(b)) for i,b in enumerate(blocks)])
    boot={n:boot_metrics(weight@np.array([co[b].sum(0) for b in blocks])) for n,co in counts.items()};paired=[];seed_deltas=[]
    for left,right,layer in PAIRS:
        for metric in ['U_bal','U_B','U_A','R_B','R_A']:
            diffs=[];rep=[]
            for seed in SEEDS:
                l=actionname(left,seed,layer);r=actionname(right,seed,layer);lv=overall[l][metric];rv=overall[r][metric];delta=lv-rv if lv is not None and rv is not None else np.nan;diffs.append(delta);rep.append(boot[l][metric]-boot[r][metric]);seed_deltas.append(dict(left=left,right=right,layer=layer,metric=metric,seed=seed,delta=delta,left_overall_met=overall[l]['target_status']=='JOINT_TARGET_MET',left_stable=stable[l],right_overall_met=overall[r]['target_status']=='JOINT_TARGET_MET',right_stable=stable[r]))
            rep=np.stack(rep);valid=np.isfinite(rep).all(0);avg=rep[:,valid].mean(0);rate=valid.mean();lo,hi=np.percentile(avg,[2.5,97.5]) if rate>=.9 else (None,None);good=np.isfinite(diffs).all();paired.append(dict(left=left,right=right,layer=layer,metric=metric,delta_mean=float(np.mean(diffs)) if good else None,delta_min=float(np.min(diffs)) if good else None,delta_max=float(np.max(diffs)) if good else None,ci_low=lo,ci_high=hi,valid_fraction=rate,valid_replicates=int(valid.sum())))
    harm=[]
    for objective in OBJECTIVES:
        for seed in SEEDS:
            name=modelname(objective,seed)+'_ALL';h=a['action_'+name]
            for reference in ['SOURCE_RF_ALL','T1_LOCAL_ALL']:
                raw=a['action_'+reference]
                for j in range(5):
                    m=np.ones(len(y),bool) if j==0 else seg==j;fix=int((m&(raw!=y)&(h==y)).sum());hurt=int((m&(raw==y)&(h!=y)).sum());harm.append(dict(method=name,objective=objective,seed=seed,reference=reference,scope='overall' if j==0 else f'segment_{j}',wrong_to_right=fix,right_to_wrong=hurt,net_corrections=fix-hurt))
    summary=[]
    for objective in OBJECTIVES:
        for layer in ['ALL','GATED']:
            names=[actionname(objective,s,layer) for s in SEEDS]
            for scope in range(5):
                vals=[overall[n] if scope==0 else next(v for v in segments if v['method']==n and v['segment']==scope) for n in names]
                for metric in ['R_B','R_A','U_B','U_A','U_bal','coverage']:
                    values=[v[metric] for v in vals];ok=all(v is not None for v in values);summary.append(dict(objective=objective,layer=layer,scope='overall' if scope==0 else f'segment_{scope}',metric=metric,mean=float(np.mean(values)) if ok else None,minimum=min(values) if ok else None,maximum=max(values) if ok else None,workpoint_seeds=sum(v['target_status']=='JOINT_TARGET_MET' for v in vals),stable_seeds=sum(stable[n] for n in names),seed_count=3,note='arithmetic metrics,not ensemble'))
    for name,rows in [('classification_comparison',[v for n,v in overall.items() if n.endswith('_ALL') or n.startswith('ALWAYS')]),('policy_comparison',[v for n,v in overall.items() if n.endswith('_GATED') or n in ['SOURCE_GATE_T1','SOURCE_RF_ALL','ALWAYS_A','ALWAYS_B']]),('segment_metrics',segments),('correction_harm',harm),('seed_summary',summary),('seed_differences',seed_deltas),('paired_intervals',paired)]:savecsv(run/f'tables/{name}.csv',rows)
    saven(run/'tables/window_counts.npz',**counts)
    d=dict(status='PILOT_COMPLETED',completed_time=now(),rows=68400,windows=684,segment_rows=[17100]*4,training_complete=9,budget=p['budget'],overall=overall,stable_workpoint=stable,segment_targets={n:[v['target_status'] for v in segments if v['method']==n] for n in overall})
    render(run,p,d,segments,summary,seed_deltas,paired,harm);writej(run/'reports/decision.json',d);writej(run/'config/evaluation_checks.json',dict(rows=68400,windows=684,segment_rows=np.bincount(seg)[1:].tolist(),blocks_per_segment=[18]*4,last_blocks=[1]*4,shared_replicates=1000,all_seeds_required_valid=True,methods=len(overall),comparisons=len(PAIRS),paired_rows=len(paired),counts_conserved=True));print('EVALUATION COMPLETE',flush=True)

def mdtable(frame,cols):
    def fmt(v):
        if v is None or isinstance(v,(float,np.floating)) and not np.isfinite(v):return 'NA'
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    return ['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']+['|'+'|'.join(fmt(v.get(c)) for c in cols)+'|' for v in frame]

def render(run,p,d,segments,summary,seed_deltas,paired,harm):
    pct=lambda v:'NA' if v is None or pd.isna(v) else f'{v*100:.4f}%'
    pp=lambda v:'NA' if v is None or pd.isna(v) else f'{v*100:.4f}'
    support=pd.read_csv(run/'calibration/support.csv').replace({np.nan:None}).to_dict('records');source=pd.read_csv(run/'source/label_support.csv').to_dict('records');val=pd.read_csv(run/'tables/source_validation.csv').replace({np.nan:None}).to_dict('records')
    lines=['# ToN source domain indicates that the training target is controlled by pilot','',f" Completed 9 training runs (3 objectives×3 fixed seeds, each 20 epochs, using the final epoch) and one evaluation of 68,400 records/684 windows;/171 windows, range [20680000,27520000), boundaries of each 17,100 {BOUNDS}。",'', 'The new target is to adapt labels to 0 and cumulative 1200; Each new model was based on T1, R200/C200, R=128B/72A, C=123B/77A. source training 50,000 clause, validation of 10,000 clause, FCE/FSC additional source family supervision, target family never read. No new methods have been developed using R2/C2 or MIXED labels.','', 'This is a developmental comparison of targeted labels and multi-cycle data evaluation, not independent 400 labels validation. The new pipeline simultaneously changes the input processing and representation; The difference in T1_LOCAL can only be attributed to the overall pipeline. The same input between the new encoders, the same backbone, and the initiated / batch / steps are used to compare training targets..1 is an engineering coefficient that does not represent a matching or optimal strength of two auxiliary targets.','', 'The method is based on [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) and [Continuous Learning for Android Malware Detection](https://www.usenix.org/conference/usenixsecurity23/presentation/chen-yizheng). This round is a self-defined source family supervision contrast objective, not a complete reproduction of security paper-level loss and proactive learning processes, nor is it intended to apply to NIDS.','', '## input, freeze and training protocol','', 'The source RF/MLP manifest specifies UNSW S_train=[0,1420000), while the rest are probability calibration, competence, test, and non-coding verification roles. Only S_train divides 10% by the PCG64(20261011) index and PCG64(20261012/13) subtracts 50k/10k; Do not use tests. Both availability_rank and original_row stability are preserved. The real Attack value is from the original CSV Attack column, which is rearranged with Label after both feature_store are encoded; The name is strip/lower only, and it is not part of the attack family. Selected records check for / family consistency.','', 'The original source is filled along, then only select source training fit 37 values mean / population standard deviation, set 1 to zero std and cut [-10,10]; 11 independent category vocabulary UNK0 embedding4 per column, total 81 dimensions. 81→128→64→32 in the bone, ReLU in the middle, BN/Dropout/ in the absence of / enhanced projection head; z=h/max(norm,1e-12), class 2 32 → 2 and family 32 → F heads are exemplified.','', 'ENC_BCE uses two logit CrossEntropyLoss; ENC_FCE plus .1 family CE; ENC_FSC plus .1 family SupCon(tau=.1). Different records from the same family are positive samples, excluding themselves and all other records, only effective anchor average, no effective anchor time zero. It’s not the same as the attack families. Adam .001 , weight_decay .0001 , batch256 , shuffle , last batch reserved , 20epoch , unadjusted / early stop . float32, which shares the initial state_dict and independent DataLoader generator batch sequences with the seeds.','',f" The real world:{p['runtime']}. All models are unified devices, TF32 restricted, no mixed accuracy. Source verification epoch0/20 is reported only and is not used to select checkpoint. The original filler comes from the original S_train, the new standardized / vocabulary with only fit selected as 50k.",'', 'Fixed R1/C1, z residual distance, k15 equal weighting, etc. rank elevation sequence after each encoder freeze; The target labels is not retransmitted..50:.05:1 and OFF of the original fixed grid with no epsilon equivalent; Each branch of C accepts ≥10 and 20 accepts ≤ error. First the maximum Ubal, then the minUBUA, then the minimum error, then the ID threshold, then the OFF threshold. C re-developed, not independent risk guaranteed.','', '## source monitoring support and quality optimization','']
    lines+=mdtable(source,['role','family','N','binary_label','known_in_train'])
    lines+=['','The following family accuracy ratios are only FCE epoch20 meaningful; Other random family heads do not provide proof of ability. Unknown family validation does not include family accuracy dividers, category 2 is still complete reporting. For each epoch, each loss, effective anchor ratios, each family is supported by a sample of source_training.csv/source_positive_support.csv.','']+mdtable(val,['objective','seed','epoch','N_B','N_A','TN','FN','FP','TP','binary_CE','balanced_accuracy','family_known','family_unknown','family_accuracy'])
    positive=pd.read_csv(run/'tables/source_positive_support.csv');positive=positive[positive.objective=='ENC_FSC'].groupby(['seed','family'],as_index=False)[['anchors','valid_anchors','positive_pairs']].sum().to_dict('records')
    lines+=['','FSC 20 wheels per family combined effective anchor supports ((normal random batch, no replacement sample):','']+mdtable(positive,['seed','family','anchors','valid_anchors','positive_pairs'])
    lines+=['','The overall anchor effective rate may be dominated by Benign. Rare attack families may not have the same batch of samples, even if they are present in the training set; This restricts actual contrast supervision, which is not the same as technical training failure, and cannot be used to claim that all families have adequate contrast training at an overall effective rate.']
    lines+=['','## targets calibration support','']+mdtable(support,['method','role','true_B','true_A','candidate_B','candidate_A','B_threshold','B_accepted','B_errors','A_threshold','A_accepted','A_errors','status'])
    lines+=['','## per seed total and four segments','', 'RB=FN/(TN+FN),RA=FP/(TP+FP),UB=TN/N_B,UA=TP/N_A; abstention is not detected, and the 0 divisor is NA. Operating point: both risks≤5%, both utilities≥10%, and each action accepts≥100 records across≥10 windows; Stabilize the = as a whole and the ≥3/4 fraction. It’s not the average production scale, but it’s the average production scale.','', '| method |RB|RA|UB|UA|Ubal| covers | overall | four phases | stable |','|---|---:|---:|---:|---:|---:|---:|---|---|---|']
    for n,v in d['overall'].items():lines.append('|'+n+'|'+'|'.join(pct(v[k]) for k in ['R_B','R_A','U_B','U_A','U_bal','coverage'])+'|'+v['target_status']+'|'+'/'.join('yes' if x=='JOINT_TARGET_MET' else 'no' for x in d['segment_targets'][n])+'|'+str(d['stable_workpoint'][n])+'|')
    lines+=['','Overall, with four complete subdividers, confusion, abstention and action windows, the 0–1 ratio is as follows:CSV is the alternative.','']+mdtable(list(d['overall'].values()),['method','N_B','N_A','TN','FN','FP','TP','abstain_B','abstain_A','accepted_B','accepted_A','windows_B','windows_A','failure_metrics'])
    lines+=['']+mdtable(segments,['method','segment','N_B','N_A','TN','FN','FP','TP','abstain_B','abstain_A','R_B','R_A','U_B','U_A','U_bal','coverage','accepted_B','accepted_A','windows_B','windows_A','target_status','failure_metrics'])
    lines+=['','## training seeds compared to scheduled','', 'For all differences in 5 indicator points per seed see seed_differences.csv; The following table lists the Ubal differences per seed (percentage points) first, followed by the three seed ranges, the arithmetic differences and the time block intervals. The average is not integrated and does not consider the seeds as three times the independent data.','']
    lines+=mdtable([dict(v,delta_pp=None if not np.isfinite(v['delta']) else 100*v['delta']) for v in seed_deltas if v['metric']=='U_bal'],['left','right','layer','seed','delta_pp','left_overall_met','left_stable'])
    lines+=['','Average, minimum and maximum indicators and the number of operating point seeds in the / sum and segment at each target / level see seed_summary.csv. The following is a summary:','']+mdtable([v for v in summary if v['scope']=='overall' and v['metric']=='U_bal'],['objective','layer','mean','minimum','maximum','workpoint_seeds','stable_seeds','seed_count'])
    lines+=['','Zone conditions: 10 window non-overlapping blocks within four sections, 1 window retained, with 18 blocks with 18 blocks drawn back, PCG64(20261014)1000 times, all targeted / seeds shared. Compute each replicate difference per seed first, then average the three seeds; The 0 of any sub-risk molecule is ineffective and does not average the remaining seeds compared to the whole of the replicated indicator; effective rate <90% range NAThe intervals are sampling only for fixed three training windows, not covering all training randomness; Point-based exploration, no multiple corrections, no guarantee of arbitrary drift.','', '| Left-right | level | indicator | minimum pp| maximum pp| equal to pp|95% range pp| effective rate |','|---|---|---|---:|---:|---:|---|---:|']
    for v in paired:lines.append(f"|{v['left']}−{v['right']}|{v['layer']}|{v['metric']}|{pp(v['delta_min'])}|{pp(v['delta_max'])}|{pp(v['delta_mean'])}|[{pp(v['ci_low'])}, {pp(v['ci_high'])}]|{v['valid_fraction']:.3f}|")
    lines+=['','## Acceptance of label correction','', 'Do not reject the door control as label correction; In the fourth paragraph, see correction_harm.csv.','']+mdtable([v for v in harm if v['scope']=='overall'],['method','reference','wrong_to_right','right_to_wrong','net_corrections'])
    conclusions=[]
    for o in OBJECTIVES:
        for layer in ['ALL','GATED']:
            vals=[v for v in summary if v['objective']==o and v['layer']==layer and v['scope']=='overall' and v['metric']=='U_bal'][0];conclusions.append(f"{o} {layer}: aggregate operating point {vals['workpoint_seeds']}/3 and operating point stable {vals['stable_seeds']}/3; Average Ubal={pct(vals['mean'])} Seed range [{pct(vals['minimum'])},{pct(vals['maximum'])}]。")
    for layer in ['ALL','GATED']:
        for l,r in [('ENC_FSC','ENC_BCE'),('ENC_FSC','ENC_FCE'),('ENC_FCE','ENC_BCE')]:
            v=next(v for v in paired if v['left']==l and v['right']==r and v['layer']==layer and v['metric']=='U_bal');conclusions.append(f"{layer} {l}−{r} Seed average: ΔUbal={pp(v['delta_mean'])} One percentage point, 95% between [{pp(v['ci_low'])},{pp(v['ci_high'])}]。")
    allstable=any(sum(d['stable_workpoint'][actionname(o,s,'GATED')] for s in SEEDS)==3 for o in OBJECTIVES)
    conclusions.append('Classification of benefits, calibration feasibility and future two-class operating point respectively. FSC − BCE contains additional family information, and only a comparison of FSC − FCE at the same time can discuss the difference in auxiliary targets; It is not possible to deduce the geometry of effective as opposed to FSC − BCE as a single unit.')
    conclusions.append('The new refers to the relative original distance as the complete input / refers to the pipeline comparison; If the FSC is not better than the normal oversight indicator, the target increase cannot be declared. 0.1 does not match intensity and does not promote optimum comparison as a target.')
    conclusions.append('Alternative independent validation is still required when there are stable results for all three seeds, combined with a source RF/T1 increase; The wheel is not expanding.' if allstable else 'All three seeds with no targets establish a stable two-class operating point and this wheel cannot be declared reliably recovered; Even if the classification improves, only the benefits of the classification are reported. The constraints of the fixed-source representation scheme do not negate all representational learning, nor do they prove that features has no information.')
    conclusions.append('This round is stopped as soon as it is completed; Do not change losses, coefficients, seeds, epoch, samples, k, R/C, or gates from evaluation outcomes. Do not automatically fine-tune, add target supervision, or start another round.')
    d['conclusions']=conclusions;lines+=['','## reading and stopping','']+['- '+v for v in conclusions]
    lines+=['','## Return and assets','',f' The actual run: `{run}`. The source index and the vocabulary, the initial /, the final model, each epoch diary, the calibration candidate rules, the freeze movements, the mating intervals and the seed count have all been preserved.','', '```bash']
    for stage in ['prepare','evaluate']:lines.append(f'{sys.executable} {run}/src/run_pilot.py --stage {stage} --run-dir {run}')
    lines+=['```',''];path=run/'reports/RESULTS_ZH.md';require(not path.exists(),'existing report')
    text='\n'.join(lines);addendum=run/'reports/interpretation_audit_ZH.md'
    if addendum.exists():text+=addendum.read_text()
    path.write_text(text)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','evaluate'],required=True);parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args();run=args.run_dir.resolve();require(run.parent==BASE/'ton_representation_pilot_v1','output scope')
    with (run/'logs/commands.sh').open('a') as f:f.write(shlex.join([sys.executable,*sys.argv])+'\n')
    try:(prepare if args.stage=='prepare' else evaluate)(run)
    except Exception as e:
        info=dict(time=now(),stage=args.stage,error=repr(e),evaluation_truth_accessed=(run/'tables/evaluation_label_access.json').exists());writej(run/'reports/failure.json',info)
        with (run/'logs/implementation_events.log').open('a') as f:f.write(str(info)+'\n')
        raise
if __name__=='__main__':main()
