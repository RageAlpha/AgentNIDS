from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent

def main():
    with (OUT/'detector_raw_platt_metrics.csv').open() as f:
        metrics=list(csv.DictReader(f))
    for r in metrics:
        for k in ['TN','FP','FN','TP']:
            r[k]=int(r[k])
    lookup={(r['domain'],r['model'],r['output']):r for r in metrics}
    data=np.load(OUT/'probability_histograms.npz')
    histograms={tuple(k.split('__')):data[k] for k in data.files}
    for family,split in [('RF','rf_Traffic'),('XGBoost','xgb_Protocol')]:
        all49=split.split('_')[0]+'_All49'
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
        for ax,(m,output) in zip(axes.flat,[(split,'Raw'),(split,'Platt'),(all49,'Raw'),(all49,'Platt')]):
            row=lookup['ToN',m,output]
            for t,(name,color) in enumerate([('Benign','#2676ad'),('Attack','#d45a35')]):
                counts=histograms[m,output][t]
                ax.stairs(counts/counts.sum(),np.linspace(0,1,51),label=f'{name} n={counts.sum():,}',color=color,linewidth=1.6)
            ax.axvline(.5,color='black',ls='--',label='threshold 0.5')
            ax.set(title=f'{m} | {output}',xlabel='Attack probability',ylabel='Within-class fraction / bin',xlim=(0,1))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'{family} / ToN: identical 0.02-wide bins; each true class sums to 1')
        fig.tight_layout();fig.savefig(OUT/'figures'/f'{family}_ToN_probability.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(9,8))
        for ax,(m,output) in zip(axes.flat,[(split,'Raw'),(split,'Platt'),(all49,'Raw'),(all49,'Platt')]):
            row=lookup['ToN',m,output]
            mat=np.array([[row['TN'],row['FP']],[row['FN'],row['TP']]])
            norm=mat/mat.sum(axis=1,keepdims=True)
            ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,f'{mat[i,j]:,}\n{100*norm[i,j]:.2f}%',ha='center',va='center',color='white' if norm[i,j]>.6 else 'black')
            ax.set(xticks=[0,1],xticklabels=['Benign','Attack'],yticks=[0,1],yticklabels=['Benign','Attack'],xlabel='Predicted (p >= 0.5)',ylabel='True',title=f'{m} | {output}')
        fig.suptitle(f'{family} / ToN confusion counts and within-true-class percentages')
        fig.tight_layout();fig.savefig(OUT/'figures'/f'{family}_ToN_confusion.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    main()
