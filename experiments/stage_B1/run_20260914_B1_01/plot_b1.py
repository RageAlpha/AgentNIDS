"""Plot compact full-data expected curves and deterministic whole-tie points."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
MODELS=['rf_Traffic','rf_All49','xgb_Protocol','xgb_All49','mlp_Behavior','mlp_All49']
LABELS=['RF-Traffic17','RF-All49','XGBoost-Protocol17','XGBoost-All49','MLP-Behavior15','MLP-All49']
COLORS=['#67a9cf','#17649a','#f5a55b','#c75014','#bc9ed1','#70459a']
OPS=list(csv.DictReader((OUT/'operating_points.csv').open()))

def main():
    handles=[Line2D([0],[0],color=c,ls='-' if m.endswith('All49') else '--',lw=2,label=l)
        for m,l,c in zip(MODELS,LABELS,COLORS)]
    for domain in ['UNSW','ToN','CIC','BoT']:
        fig,axes=plt.subplots(1,2,figsize=(13.5,5.4),sharex=True,sharey=True)
        for ax,output in zip(axes,['Raw','Platt']):
            for m,color in zip(MODELS,COLORS):
                curve=np.load(OUT/'risk_coverage_curves'/f'{domain}__{m}__{output}.npz')
                ax.plot(100*curve['expected_coverage'],100*curve['expected_risk'],color=color,
                    ls='-' if m.endswith('All49') else '--',lw=2 if m.endswith('All49') else 1.5)
                points=[r for r in OPS if r['domain']==domain and r['model']==m and r['output']==output and r['selective_risk']!='NA']
                ax.scatter([100*float(r['actual_coverage']) for r in points],[100*float(r['selective_risk']) for r in points],
                    s=24,facecolors='white',edgecolors=color,linewidths=1,zorder=3)
            ax.axhline(5,color='#555',ls=':',lw=1)
            ax.axvline(10,color='#777',ls=':',lw=1)
            ax.set(title=output,xlabel='Coverage (%)',xlim=(0,100),ylim=(0,3.1) if domain=='UNSW' else (0,102))
            ax.grid(alpha=.18)
        axes[0].set_ylabel('Accepted error rate (%)')
        fig.suptitle(f'{domain} | fixed 1 - binary entropy; classification p >= 0.5',fontsize=14)
        fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.035),frameon=False)
        fig.text(.5,.008,'Lines: expected random order within exact ties. Circles: whole-tie operating points. Empty sets have no risk marker.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.15,1,.94));fig.savefig(OUT/'figures'/f'{domain}_risk_coverage.png',dpi=180);plt.close(fig)
        # Useful counterpart: detection denominator remains ALL true Attack records.
        fig,axes=plt.subplots(1,2,figsize=(13.5,5.4),sharex=True,sharey=True)
        for ax,output in zip(axes,['Raw','Platt']):
            for m,color in zip(MODELS,COLORS):
                pts=[r for r in OPS if r['domain']==domain and r['model']==m and r['output']==output]
                x=np.array([100*float(r['actual_coverage']) for r in pts])
                z=np.array([100*float(r['effective_attack_detection']) for r in pts])
                ax.scatter(x,z,color=color,s=25,marker='o' if m.endswith('All49') else 'x')
                # Do not interpolate through unavailable whole-tie coverages.
            ax.set(title=output,xlabel='Actual whole-tie coverage (%)',xlim=(0,100),ylim=(0,102))
            ax.grid(alpha=.18)
        axes[0].set_ylabel('Correctly accepted Attack / ALL Attack (%)')
        fig.suptitle(f'{domain} | effective attack detection at all 8 specified operating points',fontsize=14)
        fig.legend(handles=[Line2D([0],[0],color=c,marker='o' if m.endswith('All49') else 'x',ls='',label=l)
            for m,l,c in zip(MODELS,LABELS,COLORS)],loc='lower center',ncol=3,bbox_to_anchor=(.5,.035),frameon=False)
        fig.text(.5,.008,'No lines between points: tied score groups can leave unattainable coverage gaps. Repeated points may overlap.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.15,1,.94));fig.savefig(OUT/'figures'/f'{domain}_effective_attack_detection.png',dpi=180);plt.close(fig)
    print('Saved 8 figures; matplotlib',matplotlib.__version__)

if __name__=='__main__': main()
