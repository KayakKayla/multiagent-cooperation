"""Export timing and learning-comparison tables and figures from local results."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    timings=[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory/'result.json').exists():continue
        result=json.loads((directory/'result.json').read_text())
        if 'warmed_median_seconds_per_update' not in result or result.get('evaluations'):continue
        metadata=json.loads((directory/'metadata.json').read_text())
        durations=result['durations_seconds'][metadata['arguments']['warmup']:]
        timings.append(dict(case=directory.name,median_seconds=result['warmed_median_seconds_per_update'],
                            minimum_seconds=min(durations)/metadata['arguments']['chunk'],
                            maximum_seconds=max(durations)/metadata['arguments']['chunk'],
                            compile_seconds=result['compile_seconds'],samples=len(durations),
                            peak_device_bytes=result['device_memory_stats'].get('peak_bytes_in_use')))
    if timings:
        with (root/'timings.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=timings[0].keys());writer.writeheader();writer.writerows(timings)
        labels={'baseline':'Current baseline','graphs_explicit_callback':'CUDA graphs; original PPO',
                'graphs_default_compact':'Graphs + compact observations','graphs_default_unroll32':'Graphs + GRU unroll 32',
                'graphs_default_unroll16':'Graphs + GRU unroll 16','graphs_minibatches32':'Graphs + 32 minibatches',
                'graphs_minibatches16':'Graphs + 16 minibatches','graphs_env512_steps128':'Graphs + 512 kitchens × 128 steps'}
        selected=[r for name in labels for r in timings if r['case']==name]
        fig,ax=plt.subplots(figsize=(10,5.7),layout='constrained')
        y=np.arange(len(selected));values=np.array([r['median_seconds'] for r in selected])
        colors=['#226b9e' if r['case'] in ('baseline','graphs_explicit_callback','graphs_default_compact') else '#d18532' for r in selected]
        ax.barh(y,values,color=colors,height=.63)
        ax.errorbar(values,y,xerr=np.array([[r['median_seconds']-r['minimum_seconds'] for r in selected],
                                            [r['maximum_seconds']-r['median_seconds'] for r in selected]]),
                    fmt='none',ecolor='#263644',capsize=3)
        ax.set_yticks(y,[labels[r['case']] for r in selected]);ax.invert_yaxis()
        for pos,value,row in zip(y,values,selected):ax.text(row['maximum_seconds']+.25,pos,f'{value:.2f} s',va='center')
        ax.set_xlim(0,max(r['maximum_seconds'] for r in selected)*1.15);ax.set_xlabel('Seconds per full-size PPO update; lower is faster')
        ax.set_title('Coordination Ring training speed\nMedian after warmup; whiskers show measured range',loc='left',pad=15)
        fig.savefig(root/'training-speed.png',dpi=170);plt.close(fig)
    curves=[]
    for name in ('baseline','graphs'):
        path=root/f'production_{name}_evaluation/evaluations.json'
        if path.exists():curves.append((name,json.loads(path.read_text())))
    if len(curves)==2:
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
        fig.suptitle('One training run per condition; 32 evaluation episodes per checkpoint',fontsize=11)
        for name,rows in curves:
            style=dict(label='Original' if name=='baseline' else 'CUDA graphs',color='#64717e' if name=='baseline' else '#087f8c',
                       marker='o' if name=='baseline' else 'x',linestyle='-' if name=='baseline' else '--',markersize=4)
            reward=[r['mean_reward'] for r in rows]
            axes[0].plot([r['update']*65536/1e6 for r in rows],reward,**style)
            axes[1].plot([r['seconds']/60 for r in rows],reward,**style)
        axes[0].set_xlabel('Training environment steps (millions)');axes[1].set_xlabel('Measured elapsed training time (minutes)')
        for ax in axes:ax.set_ylabel('Frozen-policy team reward');ax.grid(alpha=.18);ax.legend(frameon=False)
        axes[0].set_title('Effectiveness at the same training budget',loc='left')
        axes[1].set_title('Effectiveness reached sooner',loc='left')
        fig.savefig(root/'production-learning.png',dpi=170);plt.close(fig)
    partial=defaultdict(list);rows_csv=[]
    for directory in sorted(root.glob('partial_*_seed*')):
        path=directory/'result.json'
        if not path.exists():continue
        result=json.loads(path.read_text())
        prefix,seed=directory.name.rsplit('_seed',1)
        for row in result['evaluations']:
            record=dict(condition=prefix,training_seed=int(seed),update=row['update'],
                        environment_steps=row['environment_steps'],mean_reward=row['mean_reward'],mean_dishes=row['mean_dishes'],
                        training_and_startup_seconds=row['training_seconds']+row['compilation_seconds'])
            rows_csv.append(record);partial[prefix].append(record)
    if rows_csv:
        with (root/'partial-learning.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=rows_csv[0].keys());writer.writeheader();writer.writerows(rows_csv)
        fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
        labels={'partial_reference':'Original PPO + graphs','partial_mb32':'32 minibatches','partial_mb16':'16 minibatches',
                'partial_env512':'512 kitchens × 128 steps','partial_unroll16':'GRU unroll 16'}
        for prefix,records in partial.items():
            updates=sorted({r['update'] for r in records})
            means=[];low=[];high=[];times=[]
            seeds={r['training_seed'] for r in records}
            for update in updates:
                values=[r for r in records if r['update']==update]
                means.append(np.mean([r['mean_reward'] for r in values]));times.append(np.mean([r['training_and_startup_seconds']/60 for r in values]))
                low.append(min(r['mean_reward'] for r in values));high.append(max(r['mean_reward'] for r in values))
            label=labels.get(prefix,prefix.replace('partial_','').replace('_',' '))+f' (n={len(seeds)})'
            for ax,x in zip(axes,([u*65536/1e6 for u in updates],times)):
                line,=ax.plot(x,means,marker='o',label=label)
                ax.fill_between(x,low,high,color=line.get_color(),alpha=.1)
        axes[0].set_xlabel('Training environment steps (millions)')
        axes[1].set_xlabel('Training execution + startup (minutes)\nValidation time excluded')
        for ax in axes:ax.set_ylabel('Frozen-policy team reward');ax.grid(alpha=.18)
        axes[0].set_title('Partial-training effectiveness',loc='left');axes[1].set_title('Partial-training speed versus quality',loc='left')
        axes[1].legend(frameon=False,fontsize=8,loc='best')
        fig.savefig(root/'partial-learning.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(timing_cases=len(timings),partial_rows=len(rows_csv),output=str(root))))


if __name__=='__main__':main()
