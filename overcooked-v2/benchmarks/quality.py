"""Production equivalence and time-bounded, paired-seed learning comparisons."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def production_equivalence(root):
    import numpy as np
    a=json.loads((root/'production_baseline/result.json').read_text())
    b=json.loads((root/'production_graphs/result.json').read_text())
    left={x['update']:x for x in a['checkpoints']}
    right={x['update']:x for x in b['checkpoints']}
    hashes={str(step):left[step]['sha256']==right[step]['sha256'] for step in left.keys()&right.keys()}
    result={'checkpoint_weight_hash_matches':hashes,'checkpoint_steps_match':left.keys()==right.keys(),
            'all_retained_weights_identical':bool(hashes) and left.keys()==right.keys() and all(hashes.values())}
    for filename in ('state.npz','metrics.npz'):
        paths=[Path(x['run'])/'checkpoints/update-000152'/filename for x in (a,b)]
        if not all(p.exists() for p in paths):continue
        with np.load(paths[0]) as x,np.load(paths[1]) as y:
            same=x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
            result[filename+'_bitwise_identical']=same
    result['production_seconds']={name:x['production_summary']['seconds_including_compilation']
                                 for name,x in [('baseline',a),('graphs',b)]}
    result['speedup']=result['production_seconds']['baseline']/result['production_seconds']['graphs']
    (root/'production-equivalence.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('production','partial'),required=True)
    parser.add_argument('--deadline-epoch',type=float,help='Do not start a partial run that is unlikely to finish before this time')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from benchmarks.sweep import run_case
    if args.stage=='production':
        cases=[('production_baseline',ROOT/'configs/coord_ring.yaml',3600),
               ('production_graphs',ROOT/'configs/coord_ring_cuda_graphs.yaml',1800)]
        for name,config,limit in cases:
            result=run_case(args.output,name,['--config',str(config),'--max-seconds',str(limit)],
                            timeout=limit+60,program='production_run.py')
            if result['status'] not in ('complete','already_complete'):sys.exit(1)
        production_equivalence(args.output)
        for name in ('baseline','graphs'):
            run_case(args.output,'production_'+name+'_evaluation',
                     ['--run',str((args.output/('production_'+name)).resolve())],program='evaluate_saved.py')
            run_case(args.output,'production_'+name+'_final100',
                     ['--run',str((args.output/('production_'+name)).resolve()),'--episodes','100',
                      '--eval-seed','9000','--only-final'],program='evaluate_saved.py')
        return
    # All partial runs retain the original 152-update LR and shaping schedules.
    settings=[('partial_reference',{},[],'graphs_default_callback'),
              ('partial_mb32',{'host_logging':True},['--minibatches','32'],'graphs_minibatches32'),
              ('partial_env512',{'host_logging':True},['--num-envs','512','--num-steps','128'],'graphs_env512_steps128'),
              ('partial_mb16',{'host_logging':True},['--minibatches','16'],'graphs_minibatches16')]
    # Equation-preserving unrolling only merits learning runs if faster with graphs.
    recurrent=[]
    for unroll in (32,16,8,4):
        path=args.output/f'graphs_default_unroll{unroll}/result.json'
        ref=args.output/'graphs_default_callback/result.json'
        if path.exists() and ref.exists():
            if json.loads(path.read_text())['warmed_median_seconds_per_update']<.85*json.loads(ref.read_text())['warmed_median_seconds_per_update']:
                recurrent.append((json.loads(path.read_text())['warmed_median_seconds_per_update'],unroll))
    if recurrent:
        _,unroll=min(recurrent)
        settings.insert(1,(f'partial_unroll{unroll}',{'host_logging':True,'gru_unroll':unroll},[],f'graphs_default_unroll{unroll}'))
    # Complete paired conditions for seed 42 before spending time on more seeds.
    for seed in (42,43,44):
        for prefix,options,extra,timing in settings:
            name=f'{prefix}_seed{seed}'
            path=args.output/timing/'result.json'
            median=json.loads(path.read_text())['warmed_median_seconds_per_update'] if path.exists() else 6
            estimate=48*median+60
            already_complete=(args.output/name/'result.json').exists()
            if not already_complete and args.deadline_epoch and time.time()+1.25*estimate>args.deadline_epoch:
                with (args.output/'quality-skipped.jsonl').open('a') as out:
                    out.write(json.dumps(dict(case=name,reason='time budget',estimated_seconds=estimate))+'\n')
                print(json.dumps(dict(event='skipped',case=name,reason='time budget')),flush=True)
                if prefix=='partial_reference':break
                continue
            arguments=['--options',json.dumps(options),'--command-buffer','default','--stop-after','48',
                       '--eval-every','16','--episodes','32','--seed',str(seed),*extra]
            run_case(args.output,name,arguments,timeout=max(300,int(estimate*1.7)))


if __name__=='__main__':main()
