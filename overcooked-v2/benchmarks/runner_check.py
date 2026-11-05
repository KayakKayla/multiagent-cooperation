"""Check exact benchmark checkpoint/resume recovery with small PPO batches."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    common=['--num-envs','8','--num-steps','8','--minibatches','2','--epochs','1','--episodes','2',
            '--eval-every','1','--options','{"host_logging":true}','--command-buffer','default']
    cases=[('uninterrupted',['--stop-after','3']),('first',['--stop-after','2']),
           ('resumed',['--stop-after','3','--resume',str((args.output/'first/checkpoints/update-000002').resolve())])]
    for name,extra in cases:
        command=[sys.executable,str(ROOT/'benchmarks/ppo.py'),'--output',str((args.output/name).resolve()),*common,*extra]
        with (args.output/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
    from benchmarks.summarize import compare
    result=compare(args.output/'uninterrupted/probe-final.npz',args.output/'resumed/probe-final.npz')
    if not result['bitwise_equal']:raise AssertionError(result)
    (args.output/'result.json').write_text(json.dumps(dict(status='complete',agreement=result),indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
