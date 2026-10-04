"""B2 plots from saved models and compact full-data curve summaries only."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
MODELS=['rf_All49','xgb_All49','mlp_All49']
LABELS=['RF-All49','XGBoost-All49','MLP-All49']
METHODS=['one_minus_u','fu','prior']
NAMES={'one_minus_u':'Fixed 1-u','fu':'Fitted f(u)','prior':'R_fit prior'}
COLORS={'one_minus_u':'#236ba2','fu':'#d45b25','prior':'#777777'}
def rows(name):return list(csv.DictReader((OUT/name).open()))

def main():
    ops=rows('operating_points.csv')
    handles=[Line2D([0],[0],color=COLORS[m],ls=':' if m=='prior' else '-',label=NAMES[m],lw=2) for m in METHODS]
    for fold,d in [('F1','ToN'),('F2','BoT'),('F3','CIC')]:
        fig,axs=plt.subplots(3,2,figsize=(12,11),sharex=True,sharey=True)
        for i,(m,label) in enumerate(zip(MODELS,LABELS)):
            for j,o in enumerate(['Raw','Platt']):
                ax=axs[i,j]
                data=np.load(OUT/'curves'/f'{fold}__{d}__T_heldout__{m}__{o}.npz')
                for method in METHODS:
                    if method+'_risk' not in data:continue
                    ax.plot(data[method+'_coverage']*100,data[method+'_risk']*100,color=COLORS[method],
                        ls=':' if method=='prior' else '-',lw=1.6)
                    pts=[r for r in ops if r['fold']==fold and r['partition']=='T_heldout' and r['model']==m and r['output']==o and r['method']==method and r['selective_risk']!='NA']
                    ax.scatter([100*float(r['actual_coverage']) for r in pts],[100*float(r['selective_risk']) for r in pts],
                        s=22,facecolors='white',edgecolors=COLORS[method],zorder=4)
                ax.axhline(5,color='#888',ls='--',lw=.6);ax.axvline(10,color='#888',ls='--',lw=.6)
                ax.set(title=f'{label} / {o}',xlim=(0,100),ylim=(0,102));ax.grid(alpha=.15)
                if i==2:ax.set_xlabel('Coverage (%)')
                if j==0:ax.set_ylabel('Accepted error rate (%)')
        fig.suptitle(f'{fold} target {d}: unchanged base classification; entropy-only correctness model',fontsize=13)
        fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False)
        fig.text(.5,.009,'Lines: expected random order within ties. Circles: actual whole-tie operating points. No marker for empty sets.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.065,1,.97));fig.savefig(OUT/'figures'/f'{fold}_{d}_risk_coverage.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(3,2,figsize=(12,11),sharex=True,sharey=True)
        for i,(m,label) in enumerate(zip(MODELS,LABELS)):
            for j,o in enumerate(['Raw','Platt']):
                ax=axs[i,j]
                for method in METHODS:
                    pts=[r for r in ops if r['fold']==fold and r['partition']=='T_heldout' and r['model']==m and r['output']==o and r['method']==method]
                    x=[100*float(r['actual_coverage']) for r in pts]
                    for metric,marker in [('effective_attack_detection','o'),('wrong_accepted_attack_fraction','x')]:
                        ax.scatter(x,[100*float(r[metric]) for r in pts],color=COLORS[method],marker=marker,s=25)
                ax.set(title=f'{label} / {o}',xlim=(0,100),ylim=(0,102));ax.grid(alpha=.15)
                if i==2:ax.set_xlabel('Actual whole-tie coverage (%)')
                if j==0:ax.set_ylabel('Fraction of ALL Attack (%)')
        fig.suptitle(f'{fold} target {d}: correctly detected and wrongly accepted attacks',fontsize=13)
        fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False)
        fig.text(.5,.009,'Circle = correctly accepted Attack / ALL Attack; x = wrongly accepted Benign / ALL Attack. No interpolation.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.065,1,.97));fig.savefig(OUT/'figures'/f'{fold}_{d}_attack_outcomes.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey=True)
    u=np.linspace(0,1,501)
    for i,fold in enumerate(['F1','F2','F3']):
        for j,(m,label) in enumerate(zip(MODELS,LABELS)):
            ax=axs[i,j]
            for o,color in [('Raw','#236ba2'),('Platt','#d45b25')]:
                f=json.loads((OUT/'models'/f'{fold}__{m}__{o}.json').read_text())
                if f['status']!='COMPLETE':continue
                z=(u-f['anchor']-f['mean_offset'])/f['scale']
                score=1/(1+np.exp(-(f['beta0']+f['beta1']*z)))
                ax.plot(u,score,color=color,label=f'{o}: beta1={f["beta1"]:+.4f}')
                ax.axhline(f['prior'],color=color,ls=':',alpha=.5)
            ax.set(title=f'{fold} / {label}',xlim=(0,1),ylim=(0,1));ax.grid(alpha=.15);ax.legend(fontsize=8)
            if i==2:ax.set_xlabel('Normalized entropy u')
            if j==0:ax.set_ylabel('Predicted correctness f(u)')
    fig.suptitle('18 models fitted only on the corresponding two R_fit domains; dotted lines are frozen priors',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'figures'/'fitted_entropy_mappings.png',dpi=170);plt.close(fig)
    print('Seven figures written; matplotlib',matplotlib.__version__)

if __name__=='__main__':main()
