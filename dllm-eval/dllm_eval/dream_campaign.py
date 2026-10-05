"""Sequential DREAM full evaluation after verified LLaDA completion; at most six GPUs."""
import argparse,json,os,queue,signal,subprocess,sys,threading,time
from pathlib import Path
from tqdm import tqdm
from relay_cache.guards import ROOT,verify_sources,verify_environment,exclusive_lock,inventory,processes,select_idle
from relay_cache.utils import sha256,write_json
from .plan import COUNTS
from .baseline import batch_metrics,baseline_table,source_manifest
from .baseline_batch import Events,MARK
from .warmup import POLICY as MEASUREMENT

METHODS=('relay','d2cache','elastic_cache','fast_dllm_v1_no_flash')
MAX_GPUS=6
TOTAL=2*sum(COUNTS.values())


def full_proof(proof):
    if proof.get('scored')!=TOTAL:raise RuntimeError('Predecessor request count is incomplete')
    cells=proof['cells']
    if set(cells)!={f'{t}_{n}' for t in COUNTS for n in (256,512)}:raise RuntimeError('Predecessor cells changed')
    for name,cell in cells.items():
        if cell.get('samples')!=COUNTS[name.rsplit('_',1)[0]] or not cell.get('complete',True):
            raise RuntimeError('Predecessor cell is incomplete')


def llada_ready(config):
    """Completion markers alone are insufficient: require counts, successful exits and no active ranks."""
    main=Path(config['llada_output']);recovery=Path(config['llada_recovery']);v1=Path(config['llada_v1_output'])
    terminal=recovery/'exit_code'
    # The resumed output retains its original failure marker while the new
    # recovery controller is live. Only its terminal proof supersedes that history.
    if not terminal.exists():return False
    if terminal.read_text().strip()!='0':raise RuntimeError('Preceding LLaDA recovery failed')
    if not (main/'complete').exists() or not (recovery/'v1_rearmed.json').exists():
        raise RuntimeError('LLaDA recovery lacks completion/rearm proof')
    if not (main/'exit_code').exists() or (main/'exit_code').read_text().strip()!='0':
        raise RuntimeError('Missing or failed LLaDA campaign exit proof')
    state=json.loads((main/'full_summary.json').read_text())
    if state['status']!='complete' or state['active']:raise RuntimeError('LLaDA cache workers remain active')
    for method in ('d2cache','elastic_cache'):
        full_proof(json.loads((main/(method+'_full_summary.json')).read_text()))
        jobs=[j for j in state['completed'] if j['method']==method and j['phase']=='full']
        if len(jobs)!=6 or {j['rank'] for j in jobs}!=set(range(6)) or any(j['exit_code']!=0 for j in jobs):
            raise RuntimeError('LLaDA cache rank exits are incomplete')
        if any(not (main/'full'/method/f'rank{r}/complete').exists() for r in range(6)):
            raise RuntimeError('Missing LLaDA rank completion')
    marker=v1/'exit_code'
    if marker.exists() and marker.read_text().strip()!='0':raise RuntimeError('LLaDA v1 no-Flash failed')
    if not (v1/'complete').exists():return False
    if not marker.exists():raise RuntimeError('LLaDA completion lacks exit proof')
    proof=json.loads((v1/'full_summary.json').read_text());full_proof(proof)
    if proof['status']!='complete' or proof['active']:raise RuntimeError('LLaDA workers remain active')
    jobs=[j for j in proof['completed'] if j['phase']=='full']
    if len(jobs)!=6 or {j['rank'] for j in jobs}!=set(range(6)) or any(j['exit_code']!=0 for j in jobs):
        raise RuntimeError('LLaDA rank exits are incomplete')
    if any(not (v1/'full'/f'rank{r}/complete').exists() for r in range(6)):
        raise RuntimeError('Missing v1 rank completion')
    return True


def available_devices(devices,compute,active,expected):
    if len(active)>MAX_GPUS or any(type(g) is not int or g not in range(8) for g in active):
        raise RuntimeError('Six-GPU lease limit or identity changed')
    if {str(d['index']):d['uuid'] for d in devices}!=expected:raise RuntimeError('Physical GPU mapping changed')
    result=[]
    for gpu in range(8):
        if gpu in active:continue
        try:device=select_idle(devices,str(gpu),compute)
        except RuntimeError:continue
        result.append(device)
    return result[:MAX_GPUS-len(active)]


def accept(rows,row,method):
    if row['task'] not in COUNTS or row['length'] not in (256,512):raise RuntimeError('Unexpected DREAM cell')
    if row.get('method',method)!=method:raise RuntimeError('DREAM method identity changed')
    row=dict(row,method=method,id=str(row['id']))
    row.setdefault('batch_id',str((method,row['task'],row['length'],row['id'])))
    row.setdefault('batch_seconds',row['seconds']);row.setdefault('batch_size',1)
    key=(row['task'],row['length'],row['id'])
    if key in rows and rows[key]!=row:raise RuntimeError('Conflicting persisted DREAM record')
    rows[key]=row


def metrics(rows,limit=None):
    cells={}
    for task,count in COUNTS.items():
        for length in (256,512):
            part=[r for r in rows.values() if (r['task'],r['length'])==(task,length)]
            expected=limit or count
            if len(part)>expected:raise RuntimeError('Excess DREAM cell coverage')
            cells[f'{task}_{length}']=dict(batch_metrics(part),expected=expected,complete=len(part)==expected)
    return cells


def restore(folder,method):
    rows={}
    for rank in folder.glob('rank*'):
        manifest=rank/'manifest.json'
        for path in rank.glob('*/records/*.json'):
            saved=json.loads(path.read_text())
            if saved['manifest_sha256']!=sha256(manifest):raise RuntimeError('Persisted DREAM manifest changed')
            if 'assessment' not in saved:continue
            result=saved['result']
            if (saved['task'],saved['length'],str(saved['id'])) in rows:raise RuntimeError('Duplicate persisted DREAM ID')
            accept(rows,dict(task=saved['task'],length=saved['length'],id=saved['id'],
                correct=bool(saved['assessment']['correct']),seconds=result['seconds'],nfe=result['nfe'],
                **{k:result[k] for k in ('method','batch_id','batch_seconds','batch_size') if k in result}),method)
    return rows


def worker(path):
    job=json.loads(Path(path).read_text())
    if job['method']=='relay':
        from .run import parser,run_evaluation
    else:
        from .baseline_run import parser,run_evaluation
    args=parser().parse_args(job['args'])
    if args.model!='dream':raise RuntimeError('DREAM worker requires the DREAM profile')
    run_evaluation(args,Events())


def campaign(path,resume=False):
    path=Path(path).resolve();config=json.loads(path.read_text());config_hash=sha256(path)
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    if (output/'complete').exists():raise RuntimeError('Completed DREAM campaign cannot be repeated')
    if (output/'status.json').exists() and not resume:raise RuntimeError('Existing queue requires explicit --resume')
    if config['methods']!=list(METHODS) or config['max_total_gpus']!=MAX_GPUS or config['gpus']!=list(range(8)):
        raise RuntimeError('Fixed sequential method order or resource limit changed')
    if config.get('measurement')!=MEASUREMENT:raise RuntimeError('DREAM measurement policy changed')
    guard=json.loads((output/'source_guard.json').read_text())
    def check():
        if sha256(path)!=config_hash:raise RuntimeError('DREAM queue config changed')
        verify_sources(ROOT,guard)
    check();verify_environment();running={};completed=[];phase='waiting_llada';method=None
    full_progress={m:0 for m in METHODS}
    def status(state,rows=None,jobs=()):
        if len(running)>MAX_GPUS:raise RuntimeError('Six-GPU limit exceeded')
        if phase=='full' and rows is not None:full_progress[method]=len(rows)
        write_json(output/'status.json',dict(status=state,phase=phase,current_method=method,
            methods=list(METHODS),max_total_gpus=MAX_GPUS,gpus=list(range(8)),scored=sum(full_progress.values()),
            expected=TOTAL*len(METHODS),measurement=MEASUREMENT,full_progress=full_progress,stage_scored=len(rows or {}),
            stage_expected=TOTAL if phase=='full' else 8 if phase=='smoke' else 0,
            active={str(g):dict(rank=j['rank'],pid=p.pid,method=method) for g,(j,p) in running.items()},
            pending=list(jobs),completed=completed,config_sha256=config_hash,
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    with exclusive_lock(output/'campaign.lock'):
        try:
            status('waiting_llada_completion')
            print('QUEUED: all LLaDA completion -> '+ ' -> '.join(METHODS),flush=True)
            while not llada_ready(config):check();time.sleep(10)
            print('LLaDA completion barrier passed; DREAM jobs may now lease idle cards.',flush=True)
            write_json(output/'llada_barrier_passed.json',dict(status='passed',config_sha256=config_hash))
            env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'dllm-eval'),PYTHONDONTWRITEBYTECODE='1',
                PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            for name in ('CUDA_VISIBLE_DEVICES','RELAY_GPU_UUID'):env.pop(name,None)
            for method in METHODS:
                upstream=None if method=='relay' else Path(config['sources'][method])
                if upstream:source_manifest(method,upstream)
                log=Path(config['logs'][method]);log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a',encoding='utf-8',buffering=1) as stream:
                    def emit(message):
                        stream.write(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+' | '+message+'\n')
                    print('STARTING '+method+' | '+str(log),flush=True)
                    emit('DREAM Instruct / BF16 / fixed prepared prompts and scorers; two excluded startup requests per worker, then single generation.')
                    phases=[('smoke',1,1),('full',6,None)] if method.startswith('fast_dllm_v1_') else [('full',6,None)]
                    for phase,world,limit in phases:
                        folder=output/phase/method;rows=restore(folder,method);jobs=[dict(rank=r) for r in range(world)]
                        closed=set();pending=queue.Queue();rendered=set();started=time.monotonic();last=0
                        def reader(key,pipe):
                            try:
                                for line in pipe:pending.put((key,line.rstrip()))
                            finally:pipe.close();pending.put((key,None))
                        while jobs or running or len(closed)<world or not pending.empty():
                            check();devices=inventory() if jobs else [];ps=processes() if jobs else []
                            for device in available_devices(devices,ps,running,config['gpu_uuids']) if jobs else []:
                                if not jobs:break
                                gpu=device['index']
                                job=jobs.pop(0);rank=job['rank'];target=folder/f'rank{rank}'
                                args=['--model','dream','--gpu',device['uuid'],'--output',str(target),
                                    '--rank',str(rank),'--world-size',str(world),'--data-root',config['data_root'],
                                    '--hf-home',config['hf_home'],'--hf-hub-cache',config['hf_hub_cache']]
                                for entry in config['datasets']:args+=['--dataset',entry]
                                if method!='relay':args+=['--method',method,'--baseline-source',str(upstream)]
                                if limit:args+=['--limit',str(limit),'--offset','64']
                                if target.exists():args+=['--resume']
                                job_path=output/f'{phase}-{method}-{rank}.json';write_json(job_path,dict(method=method,args=args))
                                proc=subprocess.Popen([sys.executable,'-u','-m','dllm_eval.dream_campaign','--worker',str(job_path)],
                                    cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
                                running[gpu]=(job,proc);threading.Thread(target=reader,args=(rank,proc.stdout),daemon=True).start()
                                emit(f'{phase} | GPU {gpu} | rank {rank}/{world} | PID {proc.pid}')
                            try:key,line=pending.get(timeout=1)
                            except queue.Empty:key=line=None
                            if key is not None and line is None:closed.add(key)
                            if line:
                                if line.startswith(MARK):
                                    event=json.loads(line[len(MARK):])
                                    if event['kind']=='record':accept(rows,event['row'],method)
                                    elif event['kind'] in ('info','cell_start','finish'):emit(f'rank{key} | '+json.dumps(event))
                                elif 'equations=True in NormalizationConfig is deprecated' not in line:emit(f'rank{key} | '+line)
                            for gpu,(job,proc) in list(running.items()):
                                if proc.poll() is not None and job['rank'] in closed:
                                    rc=proc.wait();running.pop(gpu);completed.append(dict(method=method,phase=phase,rank=job['rank'],exit_code=rc))
                                    if rc:raise RuntimeError('DREAM worker failed; no algorithm or backend fallback')
                            if time.monotonic()-last>=10:
                                last=time.monotonic();status('running' if running else 'waiting_idle_gpus',rows,jobs)
                                stream.write(tqdm.format_meter(len(rows),8 if limit else TOTAL,time.monotonic()-started,
                                    prefix=method+' '+phase,ascii=False,ncols=125,unit='item')+'\n')
                                for name,cell in metrics(rows,limit).items():
                                    if cell['complete'] and name not in rendered:emit('Completed\n'+baseline_table({name:cell}));rendered.add(name)
                        cells=metrics(rows,limit)
                        if len(rows)!=(8 if limit else TOTAL) or not all(c['complete'] for c in cells.values()):raise RuntimeError('Incomplete DREAM coverage')
                        if any(not (folder/f'rank{r}/complete').exists() for r in range(world)):raise RuntimeError('Missing DREAM rank marker')
                        write_json(output/(phase+'_'+method+'_summary.json'),dict(status='complete',method=method,scored=len(rows),cells=cells,measurement=MEASUREMENT))
                        status('running',rows)
                        emit(phase+' complete\n'+baseline_table(cells))
                        print(method+' '+phase+' complete',flush=True)
            status('complete');(output/'complete').write_text('OK\n');(output/'exit_code').write_text('0\n')
        except BaseException:
            status('failed');(output/'exit_code').write_text('1\n')
            for _,proc in running.values():
                if proc.poll() is None:proc.send_signal(signal.SIGINT)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--worker',type=Path);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if bool(args.config)==bool(args.worker):parser.error('Specify exactly one of --config or --worker')
    worker(args.worker) if args.worker else campaign(args.config,args.resume)


if __name__=='__main__':main()
