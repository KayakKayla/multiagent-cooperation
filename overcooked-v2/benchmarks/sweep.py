"""Run isolated GPU cases serially with timeouts and lightweight telemetry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def run_case(root, name, arguments, timeout=360, program='ppo.py'):
    directory = root/name
    if (directory/'result.json').exists():
        previous=json.loads((directory/'result.json').read_text())
        return {'case':name,'status':'already_complete' if previous.get('status')=='complete' else 'existing_failed'}
    log = root/(name+'.log')
    command = [sys.executable,'-u',str(ROOT/'benchmarks'/program),'--output',str(directory),*arguments]
    stop = threading.Event()
    def monitor():
        with (root/(name+'-gpu.csv')).open('w') as out:
            while not stop.is_set():
                try:
                    row = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
                        '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
                        '--format=csv,noheader,nounits'],text=True,timeout=5)
                    out.write(row); out.flush()
                except (OSError,subprocess.SubprocessError):
                    pass
                stop.wait(2)
    thread = threading.Thread(target=monitor,daemon=True); thread.start()
    started=time.monotonic()
    print(json.dumps({'event':'start','case':name,'command':command}),flush=True)
    with log.open('w') as output:
        process=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=timeout)
            status='complete' if code==0 else 'failed'
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
            code=process.returncode; status='timeout'
    stop.set();thread.join(timeout=6)
    row=dict(case=name,status=status,exit_code=code,wall_seconds=time.monotonic()-started,command=command)
    if (directory/'result.json').exists():
        result=json.loads((directory/'result.json').read_text())
        row.update(median_seconds=result.get('warmed_median_seconds_per_update'),compile_seconds=result.get('compile_seconds'))
    with (root/'sweep.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
    print(json.dumps({'event':'finished',**row}),flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--group',choices=('systems','tuning','extras','refine','all','remaining'),default='systems')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.group in ('all','remaining'):
        groups=['systems','refine','tuning','extras'] if args.group=='all' else ['refine','tuning','extras']
        for group in groups:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(args.output.resolve()),
                            '--group',group],cwd=ROOT,check=True)
        return
    host={'host_logging':True}
    def variant(options,extra=()):return ['--options',json.dumps(options),*extra]
    if args.group=='systems':
        cases=[
            ('host_logging',variant(host)),
            ('conditional_reset',variant({**host,'conditional_reset':True})),
            ('compact_obs',variant({**host,'compact_obs':True})),
            ('fused_gru',variant({**host,'fused_gru':True})),
            ('gru_unroll4',variant({**host,'gru_unroll':4})),
            ('donate',variant({**host,'donate':True})),
            ('preallocate60',variant(host,['--preallocate','true'])),
            ('equivalent_combined',variant({**host,'conditional_reset':True,'compact_obs':True,'donate':True})),
            ('graphs_fusion',variant(host,['--command-buffer','FUSION'])),
            ('graphs_fusion_custom',variant(host,['--command-buffer','FUSION,CUSTOM_CALL'])),
            ('tf32',variant(host,['--precision','tensorfloat32'])),
            ('baseline_repeat',[]),
        ]
    elif args.group=='tuning':
        test_env={**os.environ,'JAX_PLATFORMS':'cuda,cpu','XLA_FLAGS':'--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true',
                  'XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
        subprocess.run([sys.executable,'-m','unittest','tests.test_ppo_performance','-v'],cwd=ROOT,env=test_env,check=True)
        cases=[('graphs_default_callback',['--command-buffer','default']),
               ('graphs_explicit_callback',['--command-buffer','FUSION,CUBLAS,CUSTOM_CALL,CUDNN']),
               ('graphs_default_compact',variant({**host,'compact_obs':True},['--command-buffer','default'])),
               ('graphs_default_unroll4',variant({**host,'gru_unroll':4},['--command-buffer','default'])),
               ('graphs_default_unroll8',variant({**host,'gru_unroll':8},['--command-buffer','default'])),
               ('graphs_minibatches32',variant(host,['--minibatches','32','--command-buffer','default'])),
               ('graphs_minibatches16',variant(host,['--minibatches','16','--command-buffer','default'])),
               ('graphs_env512_steps128',variant(host,['--num-envs','512','--num-steps','128','--command-buffer','default'])),
               ('graphs_bf16_cnn',variant({**host,'bf16_cnn':True},['--command-buffer','default'])),
               ('graphs_nondeterministic',variant(host,['--no-deterministic','--command-buffer','default']))]
    elif args.group=='refine':
        cases=[('graphs_fusion_callback',['--command-buffer','FUSION']),
               ('graphs_default',variant(host,['--command-buffer','default'])),
               ('graphs_fusion_compact',variant({**host,'compact_obs':True},['--command-buffer','FUSION'])),
               ('graphs_fusion_unroll4',variant({**host,'gru_unroll':4},['--command-buffer','FUSION'])),
               ('graphs_fusion_fused_unroll4',variant({**host,'fused_gru':True,'gru_unroll':4},['--command-buffer','FUSION'])),
               ('gru_unroll8',variant({**host,'gru_unroll':8})),
               ('gru_unroll16',variant({**host,'gru_unroll':16})),
               ('fused_unroll4',variant({**host,'fused_gru':True,'gru_unroll':4})),
               ('fused_unroll8',variant({**host,'fused_gru':True,'gru_unroll':8})),
               ('triton_gemm',variant(host,['--extra-xla=--xla_gpu_triton_gemm_any=true'])),
               ('latency_scheduler',variant(host,['--extra-xla=--xla_gpu_enable_latency_hiding_scheduler=true']))]
    else:
        for unroll in (16,32):
            run_case(args.output,f'graphs_default_unroll{unroll}',
                     variant({**host,'gru_unroll':unroll},['--command-buffer','default']))
        run_case(args.output,'graphs_nondeterministic_retry',
                 variant(host,['--no-deterministic','--command-buffer','default']))
        run_case(args.output,'checkpoint_cost',[],program='checkpoint_cost.py')
        for name,options in [('phases_baseline',{}),('phases_reset',{'conditional_reset':True}),
                             ('phases_fused',{'fused_gru':True})]:
            run_case(args.output,name,['--options',json.dumps(options)],program='phases.py')
        for name,options in [('cache_callback_cold',{}),('cache_callback_warm',{}),
                             ('cache_host_cold',host),('cache_host_warm',host)]:
            cache_token=hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:12]
            run_case(args.output,name,variant(options,['--warmup','0','--samples','1','--cache-dir',
                '/tmp/se390-ppo-cache-'+cache_token+'-'+('host' if options else 'callback')]))
        run_case(args.output,'cpu_baseline',['--backend','cpu','--warmup','1','--samples','2'],timeout=420)
        run_case(args.output,'two_seed_batch',[],timeout=360,program='seed_batch.py')
        run_case(args.output,'runner_check',[],timeout=390,program='runner_check.py')
        return
    for name,arguments in cases:
        run_case(args.output,name,arguments,timeout=240 if name.startswith('graphs') else 420)


if __name__=='__main__':main()
