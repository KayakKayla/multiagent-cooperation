"""Measure the unmodified production launcher and retain periodic policy weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/coord_ring.yaml')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-seconds',type=int,default=4200)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    command=['bash',str(ROOT/'run.sh'),'train','--config',str(args.config.resolve()),'--seed',str(args.seed)]
    started=time.monotonic();stop=threading.Event();timed_out=threading.Event()
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                             bufsize=1,start_new_session=True)
    def watchdog():
        if not stop.wait(args.max_seconds):
            timed_out.set();os.killpg(process.pid,signal.SIGTERM)
            if not stop.wait(10):os.killpg(process.pid,signal.SIGKILL)
    threading.Thread(target=watchdog,daemon=True).start()
    run=None;events=[];checkpoints=[]
    with (args.output/'training.log').open('w') as log:
        for line in process.stdout:
            elapsed=time.monotonic()-started
            log.write(line);log.flush()
            match=re.search(r'Results: (.+)\. Checkpoint every',line)
            if match:
                run=Path(match.group(1))
                shutil.copyfile(run/'config.json',args.output/'config.json')
            if re.match(r'Update \d+/',line) or line.startswith('Saved checkpoint'):
                event=dict(seconds=elapsed,message=line.strip());events.append(event)
                with (args.output/'events.jsonl').open('a') as out:out.write(json.dumps(event)+'\n')
                print(json.dumps(event),flush=True)
            match=re.match(r'Saved checkpoint (update-\d+)',line)
            if match and run:
                source=run/'checkpoints'/match.group(1)/'params.msgpack'
                dest=args.output/(match.group(1)+'.msgpack')
                shutil.copyfile(source,dest)
                checkpoints.append(dict(update=int(match.group(1).split('-')[1]),file=dest.name,
                                        seconds=elapsed,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
                (args.output/'policies.json').write_text(json.dumps(checkpoints,indent=2))
    code=process.wait();stop.set()
    result=dict(status='timeout' if timed_out.is_set() else ('complete' if code==0 else 'failed'),
                exit_code=code,command=command,run=str(run) if run else None,
                wall_seconds=time.monotonic()-started,checkpoints=checkpoints)
    if run and (run/'summary.json').exists():
        result['production_summary']=json.loads((run/'summary.json').read_text())
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if code:sys.exit(code)


if __name__=='__main__':main()
