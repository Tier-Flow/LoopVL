"""Persistently schedule a configurable maximum of inference workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ORDER=['RealWorldQA','MMStar','ChartQA_TEST','MMMU-Pro','MMEval-Pro','MathVision','AI2D_TEST','POPE','MMK12','VisualPuzzles','VMCBench_DEV','VisuLogic','HallusionBench','LogicVista','BabyVision','MathVision-WildPhoto']
COUNTS={'AI2D_TEST':3088,'BabyVision':388,'ChartQA_TEST':2500,'HallusionBench':951,'LogicVista':447,'MMEval-Pro':6414,'MMK12':2000,'MMMU-Pro':1730,'MMStar':1500,'MathVision':3040,'MathVision-WildPhoto':304,'POPE':5127,'RealWorldQA':765,'VMCBench_DEV':1000,'VisuLogic':1000,'VisualPuzzles':1168}

def atomic(path,value):
    temporary=path.with_name(path.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def read_json(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def gpu_inventory():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
    devices={}
    for line in output.splitlines():
        index,uuid,name,memory=[field.strip() for field in line.split(',',3)]
        devices[index]={'uuid':uuid,'name':name,'memory_mib':int(memory)}
    return devices

def choose_gpu(gpus,active,workers_per_gpu):
    loads={gpu:sum(entry['physical_gpu']==gpu for entry in active.values()) for gpu in gpus}
    available=[gpu for gpu in gpus if loads[gpu]<workers_per_gpu]
    return min(available,key=lambda gpu:loads[gpu]) if available else None

def isolated_worker_env(base,gpu_uuid):
    env=base.copy()
    env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    env['CUDA_VISIBLE_DEVICES']=gpu_uuid
    return env

def validate_existing_outputs(root):
    for name in ORDER:
        path=root/'results'/name/'predictions.jsonl'
        if not path.exists():continue
        data=path.read_bytes()
        if data and not data.endswith(b'\n'):
            raise ValueError(f'Interrupted JSONL tail needs explicit backup/recovery before resume: {path}')
        for number,line in enumerate(data.splitlines(),1):
            if line.strip():
                try:json.loads(line)
                except ValueError as error:raise ValueError(f'Invalid JSONL {path}:{number}') from error

def main():
    package=Path(__file__).resolve().parent
    repo=package.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=repo/'outputs/benchmarks')
    parser.add_argument('--benchmark-root',type=Path,default=package)
    parser.add_argument('--asset-root',type=Path,default=repo/'data/server_snapshot_20260906')
    parser.add_argument('--model-dir',type=Path,default=repo/'model')
    parser.add_argument('--dry-run',action='store_true',help='Print paths and task counts without loading CUDA or writing files')
    parser.add_argument('--poll-seconds',type=int,default=15)
    parser.add_argument('--max-workers',type=int,default=None)
    parser.add_argument('--gpus',default='0',help='Comma-separated nvidia-smi GPU indices')
    parser.add_argument('--workers-per-gpu',type=int,choices=[1,2,3,4],default=4)
    parser.add_argument('--launch-interval-seconds',type=float,default=2.0)
    args=parser.parse_args()
    for field in ('root','benchmark_root','asset_root','model_dir'):
        path=getattr(args,field)
        setattr(args,field,(path if path.is_absolute() else repo/path).resolve())
    gpus=[gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
    if args.dry_run:
        print(json.dumps({'model_dir':str(args.model_dir),'asset_root':str(args.asset_root),
            'output_root':str(args.root),'datasets':COUNTS,'total_predictions':sum(COUNTS.values()),
            'gpus':gpus,'workers_per_gpu':args.workers_per_gpu,'prompt_policy':'original-brief'},indent=2))
        return
    if os.name!='posix':parser.error('GPU scheduling requires Linux (fcntl and CUDA); CPU scoring is cross-platform')
    import fcntl
    for protected in (args.benchmark_root,args.model_dir,args.asset_root):
        if args.root==protected or protected in args.root.parents:
            parser.error(f'Output cannot be inside protected inputs: {protected}')
    if not (args.model_dir/'model.safetensors').is_file():
        parser.error('Download the complete LoopVL model snapshot into model/ before inference')
    inventory=gpu_inventory()
    if not gpus or len(set(gpus))!=len(gpus) or any(gpu not in inventory for gpu in gpus):
        parser.error('Select unique available GPU indices')
    capacity=len(gpus)*args.workers_per_gpu
    if args.max_workers is None:args.max_workers=capacity
    if not 1<=args.max_workers<=capacity:parser.error('--max-workers exceeds per-GPU capacity')
    if args.launch_interval_seconds<0:parser.error('--launch-interval-seconds must be nonnegative')
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A supervisor or its active workers already holds the run lock')
    validate_existing_outputs(root)
    # Verify data against pinned hashes before scheduling. Downloads are a separate explicit step.
    subprocess.run([sys.executable,str(package/'prepare.py'),'--asset-root',str(args.asset_root.parent),
        '--status-root',str(root/'inputs_status'),'--verify-only'],check=True)
    placement={'gpus':{gpu:inventory[gpu] for gpu in gpus},'workers_per_gpu':args.workers_per_gpu,
               'maximum_simultaneous_inference_processes':args.max_workers,'worker_logical_device':'cuda:0'}
    atomic(root/'supervisor_identity.json',{'pid':os.getpid(),'started_at':time.time(),**placement})
    for name in ['logs','results','comparison','inputs_status']:(root/name).mkdir(exist_ok=True)
    state=read_json(root/'scheduler_state.json',{})
    attempts=state.get('attempts',{})
    execution_failures=state.get('execution_failures',{})
    completed={name for name in state.get('completed',[]) if name in COUNTS
        and read_json(root/'results'/name/'summary.json',{}).get('full_benchmark_complete')
        and read_json(root/'results'/name/'summary.json',{}).get('completed')==COUNTS[name]}
    failed=dict(state.get('failed',{}));active={}
    interrupted=False
    def signal_stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    env=os.environ.copy()
    for key in ['MODELSCOPE_API_TOKEN','HF_TOKEN']:env.pop(key,None)
    env.update(PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    def collect():
        cmd=[sys.executable,str(package/'collect.py'),'--benchmark-root',str(args.benchmark_root),'--results-root',str(root/'results'),'--output-dir',str(root/'comparison')]
        try:
            with (root/'logs/collection.log').open('a') as log:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=log,timeout=180)
            if result.returncode:print('COLLECTION_ERROR',result.returncode,flush=True)
        except (OSError,subprocess.TimeoutExpired) as error:
            print('COLLECTION_ERROR',str(error),flush=True)
    def status():
        datasets={}
        for name in ORDER:
            progress=read_json(root/'results'/name/'progress.json',{})
            summary=read_json(root/'results'/name/'summary.json',{})
            datasets[name]={'expected':COUNTS[name],'progress':progress,'summary':summary,
                            'state':'complete' if name in completed else 'failed' if name in failed else 'running' if name in active else 'queued'}
        atomic(root/'scheduler_state.json',{'pid':os.getpid(),'updated_at':time.time(),'maximum_simultaneous_inference_processes':args.max_workers,
            'attempts':attempts,'execution_failures':execution_failures,'placement':placement,'completed':sorted(completed),'failed':failed,
            'active':{name:{'pid':entry['process'].pid,'started_at':entry['started_at'],
                            'physical_gpu':entry['physical_gpu'],'gpu_uuid':entry['gpu_uuid']} for name,entry in active.items()},
            'datasets':datasets,'input_download':read_json(root/'inputs_status/download_status.json',{}),
            'state':'stopping' if interrupted else 'complete' if len(completed)==16 else 'finished_with_errors' if len(completed)+len(failed)==16 else 'running'})
    collect()
    try:
        while len(completed)+len(failed)<len(ORDER) and not interrupted:
            changed=False
            for name,entry in list(active.items()):
                code=entry['process'].poll()
                if code is None:continue
                entry['log'].close();del active[name];changed=True
                summary=read_json(root/'results'/name/'summary.json',{})
                amount=summary.get('completed',summary.get('total',0))
                if code==0 and amount==COUNTS[name] and summary.get('full_benchmark_complete') and not summary.get('unresolved_error_count'):
                    completed.add(name)
                    print('COMPLETE',name,amount,flush=True)
                else:
                    execution_failures[name]=execution_failures.get(name,0)+1
                    if execution_failures[name]>=3:
                        failed[name]={'exit_code':code,'last_summary':summary,'reason':'Three observed execution failures; inspect dataset log'}
                        print('FAILED',name,code,flush=True)
                    else:print('RETRY_PENDING',name,code,flush=True)
            for name in ORDER:
                if interrupted:break
                if len(active)>=args.max_workers:break
                if name in completed or name in failed or name in active:continue
                readiness=read_json(root/'inputs_status'/(name+'.ready.json'),{})
                if not readiness.get('ready'):continue
                gpu=choose_gpu(gpus,active,args.workers_per_gpu)
                if gpu is None:break
                attempts[name]=attempts.get(name,0)+1
                log=(root/'logs'/(name+'.log')).open('a',buffering=1)
                command=[sys.executable,'-u',str(package/'worker.py'),'--dataset',name,'--benchmark-root',str(args.benchmark_root),
                    '--asset-root',str(args.asset_root),'--model-dir',str(args.model_dir),'--output-root',str(root/'results'),
                    '--prompt-policy','original-brief','--device','cuda:0']
                worker_env=isolated_worker_env(env,inventory[gpu]['uuid'])
                process=subprocess.Popen(command,env=worker_env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,pass_fds=(lock.fileno(),))
                entry={'process':process,'log':log,'started_at':time.time(),'physical_gpu':gpu,'gpu_uuid':inventory[gpu]['uuid']}
                active[name]=entry;changed=True
                with (root/'placement_history.jsonl').open('a') as history:
                    history.write(json.dumps({'dataset':name,'pid':process.pid,'started_at':entry['started_at'],
                        'physical_gpu':gpu,'gpu_uuid':entry['gpu_uuid'],'worker_logical_device':'cuda:0',
                        'launch_attempt':attempts[name],'worker_command':command})+'\n')
                print('START',name,'PID',process.pid,'GPU',gpu,'active',len(active),flush=True)
                time.sleep(args.launch_interval_seconds)
            status()
            if changed:collect()
            time.sleep(args.poll_seconds)
        if not interrupted:
            # All pending tasks must be accounted for. In-flight tasks finish before final reporting.
            collect();status()
            report=read_json(root/'comparison/comparison.json',{})
            final={'finished_at':time.time(),'completed_datasets':len(completed),'failed':failed,'full_integrity_passed':report.get('full_integrity_passed',False),
                   'success':len(completed)==16 and report.get('all_complete',False) and report.get('full_integrity_passed',False),'comparison':str(root/'comparison/comparison.md')}
            atomic(root/'FINAL_STATUS.json',final)
            print('FINAL',json.dumps(final),flush=True)
    finally:
        if interrupted:
            for entry in active.values():entry['process'].terminate()
            for entry in active.values():
                try:entry['process'].wait(timeout=30)
                except subprocess.TimeoutExpired:entry['process'].kill()
                entry['log'].close()
            status()

if __name__=='__main__':main()
