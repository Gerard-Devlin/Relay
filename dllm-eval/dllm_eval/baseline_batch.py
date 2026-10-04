"""Frozen-source multi-GPU campaign: one log per algorithm, then full evaluation."""
import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from tqdm import tqdm
from relay_cache.guards import ROOT, inventory, processes, select_idle, exclusive_lock
from relay_cache.utils import write_json, sha256
from .baseline import SOURCES, source_manifest, settings, batch_metrics, baseline_table
from .plan import COUNTS

MARK = '@@BASELINE_EVENT '
METHOD_ORDER = ('d2cache', 'elastic_cache')


def method_log_paths(log_dir, output):
    return {method: Path(log_dir) / f'{method}_{Path(output).name}.log' for method in METHOD_ORDER}


class MethodLogs:
    """One transcript per algorithm, shared by all of that algorithm's workers."""
    def __init__(self, paths, resume=False):
        self.paths, self.resume, self.streams = paths, resume, {}

    def __enter__(self):
        try:
            for method, path in self.paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                self.streams[method] = path.open('a' if self.resume else 'x', encoding='utf-8', buffering=1)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def write(self, text, method=None):
        streams = self.streams.values() if method is None else (self.streams[method],)
        for stream in streams:
            stream.write(text)

    def __exit__(self, *_):
        for stream in self.streams.values(): stream.close()


def stage_jobs(phase, world):
    return [dict(method=method, rank=rank) for method in METHOD_ORDER for rank in range(world)]


def current_method(jobs, running):
    remaining = {job['method'] for job in jobs}
    remaining.update(job['method'] for job, _ in running.values())
    return next((method for method in METHOD_ORDER if method in remaining), None)


def restored_rows(output, phase):
    """Reuse persisted assessments; workers still validate each prompt on resume."""
    rows = {}
    for method in SOURCES:
        for rank in (output / phase / method).glob('rank*'):
            manifest = rank / 'manifest.json'
            for path in rank.glob('*/records/*.json'):
                row = json.loads(path.read_text())
                if not manifest.is_file() or row['manifest_sha256'] != sha256(manifest):
                    raise RuntimeError('Saved generation manifest changed: ' + str(path))
                if row['result']['method'] != method:
                    raise RuntimeError('Saved baseline method changed')
                if 'assessment' not in row:
                    continue
                result = row['result']
                scalar = dict(task=row['task'],length=row['length'],id=row['id'],
                    correct=bool(row['assessment']['correct']),seconds=result['seconds'],
                    nfe=result['nfe'],method=method,batch_id=result['batch_id'],
                    batch_seconds=result['batch_seconds'],batch_size=result['batch_size'])
                key = (method, row['task'], row['length'], row['id'])
                if key in rows:
                    raise RuntimeError('Duplicate persisted baseline request')
                rows[key] = scalar
    return rows


class Events:
    def send(self, kind, **payload):
        print('\n'+MARK+json.dumps(dict(kind=kind, **payload)), flush=True)
    def info(self, message):self.send('info', message=message)
    def start_run(self, args, samples):self.send('run', total=sum(map(len,samples.values()))*len(args.lengths))
    def start_cell(self, task, length, total):self.send('cell_start', task=task, length=length, total=total)
    def update(self, row):self.send('record', row=row)
    def finish_cell(self):self.send('cell_end')
    def finish(self, cells, output):self.send('finish', cells=cells, output=str(output))


def worker(job_path):
    from .baseline_run import parser, run_evaluation
    job = json.loads(Path(job_path).read_text())
    args = parser().parse_args(job['args'])
    run_evaluation(args, Events())


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', nargs='+', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--dataset', action='append', default=[])
    p.add_argument('--hf-home', type=Path)
    p.add_argument('--hf-hub-cache', type=Path)
    p.add_argument('--log-dir', type=Path, default=ROOT/'log')
    p.add_argument('--worker', type=Path)
    p.add_argument('--resume', action='store_true', help='Resume the frozen campaign and append its single log')
    return p


def freeze(output):
    target=output/'source'
    target.mkdir()
    for folder in ('relay_cache','dllm-eval/dllm_eval','dllm-eval/configs'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json'):
                dest=target/p.relative_to(ROOT)
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,dest)
                dest.chmod(0o444)
    return target


def aggregate(rows, phase):
    cells={}
    for method in SOURCES:
        cells[method]={}
        for length in (256,512):
            for task,count in COUNTS.items():
                part=[r for r in rows.values() if r['method']==method and r['task']==task and r['length']==length]
                n=len(part);expected=2 if phase=='smoke' else count
                cells[method][f'{task}_{length}']=dict(batch_metrics(part),expected=expected,complete=n==expected)
    return cells


def campaign(args):
    if len(set(args.gpus))!=len(args.gpus) or not args.gpus:
        raise ValueError('Distinct physical GPUs required')
    if any(index < 0 for index in args.gpus):
        raise ValueError('Physical GPU indices must be nonnegative')
    args.output=args.output.resolve();args.upstream_root=args.upstream_root.resolve()
    if args.resume:
        previous=json.loads((args.output/'launch.json').read_text())
        if (args.output/'complete').exists():
            raise RuntimeError('Campaign already complete; duplicate run refused')
        source=(args.output/'source').resolve()
        if previous['gpus']!=args.gpus or Path(previous['source']).resolve()!=source or not source.is_dir():
            raise RuntimeError('Frozen campaign resources or source path changed')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        source=freeze(args.output)
    upstream={method:source_manifest(method,args.upstream_root/method) for method in SOURCES}
    args.log_dir.mkdir(parents=True,exist_ok=True)
    logs=method_log_paths(args.log_dir,args.output)
    log=logs[METHOD_ORDER[0]]
    info=dict(output=str(args.output),log=str(log),logs={m:str(p) for m,p in logs.items()},source=str(source),gpus=args.gpus,profiles={m:settings(m) for m in SOURCES},upstream=upstream)
    if args.resume:
        if previous['upstream']!=upstream or previous['profiles']!=info['profiles'] or previous.get('logs')!=info['logs']:
            raise RuntimeError('Frozen campaign upstream, settings or log changed')
        write_json(args.output/'resume_scheduler.json',dict(method_order=METHOD_ORDER,resumed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    else:
        write_json(args.output/'launch.json',info)
    write_json(ROOT/'runs/latest_baselines.json',info)
    env=os.environ.copy()
    for name in ('CUDA_VISIBLE_DEVICES','RELAY_GPU_UUID'):env.pop(name,None)
    env.update(PYTHONPATH=str(source)+os.pathsep+str(source/'dllm-eval'),PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    common=['--data-root',str(args.data_root)]
    for entry in args.dataset:common+=['--dataset',entry]
    for value,name in ((args.hf_home,'--hf-home'),(args.hf_hub_cache,'--hf-hub-cache')):
        if value:common+=[name,str(value)]
    with exclusive_lock(args.output/'campaign.lock'),MethodLogs(logs,args.resume) as logfile:
        if args.resume and (args.output/'exit_code').exists():
            write_json(args.output/'previous_scheduler_exit.json',dict(exit_code=(args.output/'exit_code').read_text().strip(),reason='User requested sequential six-GPU scheduling'))
            (args.output/'exit_code').unlink()
        def emit(message,method=None):
            text=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+' | '+message
            logfile.write(text+'\n',method);tqdm.write(text,file=sys.stderr)
        emit('Pinned LLaDA-8B-Instruct / BF16 / prompts / final-expression-v3 + official code execution')
        emit('d2Cache: threshold=.90, block32, generation sigma=0, cache sigma=10/inflate4.','d2cache')
        emit('Elastic-Cache: window16, threshold/gamma=.90, track1, official EOS stop.','elastic_cache')
        emit('Warmed generation time: batch latency and amortized time/item are separate; no LLaDA/v1/Relay reruns.')
        emit('Six-GPU method order: d2Cache first, then Elastic-Cache. Completed records are reused on resume.')
        try:
            for phase in ('smoke','full'):
                if args.resume and phase=='smoke':
                    smoke=json.loads((args.output/'smoke_summary.json').read_text())
                    if smoke['scored']!=32 or not all(c['complete'] for cells in smoke['cells'].values() for c in cells.values()):
                        raise RuntimeError('Completed smoke checks required to resume full evaluation')
                    emit('Existing 32/32 smoke checks retained; no duplicate smoke generation.')
                    continue
                world=1 if phase=='smoke' else len(args.gpus)
                total=32 if phase=='smoke' else 2*2*sum(COUNTS.values())
                jobs=stage_jobs(phase,world)
                running={};completed=[];pending=queue.Queue();rows=restored_rows(args.output,phase) if args.resume else {};rendered=set();closed=set();started=time.monotonic();last_status=0;last_wait=0
                emit(f'{phase}: {len(rows)} previously scored records restored.')
                def reader(key,stream):
                    try:
                        for line in stream:pending.put((key,line.rstrip('\r\n')))
                    finally:stream.close();pending.put((key,None))
                def status(state):
                    report=dict(status=state,phase=phase,scored=len(rows),expected=total,cells=aggregate(rows,phase),
                        current_method=current_method(jobs,running),
                        active={str(gpu):dict(method=j['method'],rank=j['rank'],pid=p.pid) for gpu,(j,p) in running.items()},
                        completed=completed,pending=jobs,log=str(logs.get(current_method(jobs,running),log)),logs=info['logs'],updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                    write_json(args.output/'status.json',report)
                    return report
                bar=tqdm(total=total,initial=len(rows),desc=f'Baselines {phase} | GPU '+','.join(map(str,args.gpus)),unit='req',ascii=False,dynamic_ncols=True,file=sys.stderr)
                try:
                    while jobs or running or len(closed)<2*world or not pending.empty():
                        inv=inventory() if jobs else [];ps=processes() if jobs else []
                        for gpu in args.gpus:
                            if gpu in running or not jobs:continue
                            if jobs[0]['method']!=current_method(jobs,running):
                                continue
                            try:device=select_idle(inv,str(gpu),ps)
                            except RuntimeError:continue
                            job=jobs.pop(0);method=job['method'];rank=job['rank']
                            output=args.output/phase/method/f'rank{rank}'
                            cmd_args=common+['--method',method,'--baseline-source',str(args.upstream_root/method),
                                '--gpu',device['uuid'],'--output',str(output),'--rank',str(rank),'--world-size',str(world)]
                            if phase=='smoke':cmd_args+=['--limit','2']
                            if args.resume and output.exists():cmd_args+=['--resume']
                            job_path=args.output/f'{phase}-{method}-rank{rank}.json';write_json(job_path,dict(args=cmd_args,gpu=gpu))
                            key=f'{phase}-{method}-{rank}'
                            proc=subprocess.Popen([sys.executable,'-u','-m','dllm_eval.baseline_batch','--worker',str(job_path)],
                                cwd=source,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
                            running[gpu]=(job,proc);threading.Thread(target=reader,args=(key,proc.stdout),daemon=True).start()
                            emit(f'{phase} | {method} | GPU {gpu} | rank {rank}/{world} | PID {proc.pid}',method)
                        try:key,line=pending.get(timeout=1)
                        except queue.Empty:key=line=None
                        if key is not None and line is None:closed.add(key)
                        if line:
                            if line.startswith(MARK):
                                event=json.loads(line[len(MARK):])
                                if event['kind']=='record':
                                    row=event['row'];ident=(row['method'],row['task'],row['length'],row['id'])
                                    if ident in rows:
                                        if not args.resume or rows[ident]!=row:
                                            raise RuntimeError('Duplicate or changed baseline request')
                                    else:
                                        rows[ident]=row;bar.update(1)
                                elif event['kind'] in ('info','cell_start','finish'):emit(key+' | '+json.dumps(event),next(m for m in METHOD_ORDER if key.startswith(phase+'-'+m+'-')))
                            else:emit(key+' | '+line.strip(),next(m for m in METHOD_ORDER if key.startswith(phase+'-'+m+'-')))
                        for gpu,(job,proc) in list(running.items()):
                            key=f'{phase}-{job["method"]}-{job["rank"]}'
                            # Drain the entire worker stream before handling its exit.
                            # Otherwise a fast failure can truncate its traceback.
                            if proc.poll() is not None and key in closed:
                                code=proc.wait();completed.append(dict(**job,gpu=gpu,exit_code=code));running.pop(gpu)
                                emit(f'{phase} | {job["method"]} | GPU {gpu}: exit={code}',job['method'])
                                if code:raise RuntimeError('Baseline worker failed; all persisted records retained')
                        if time.monotonic()-last_status>=5:
                            report=status('running');last_status=time.monotonic()
                            method=report['current_method']
                            if method is not None:
                                n=sum(row['method']==method for row in rows.values())
                                method_total=16 if phase=='smoke' else 2*sum(COUNTS.values())
                                logfile.write(tqdm.format_meter(n,method_total,time.monotonic()-started,prefix=method+' '+phase,ascii=False,ncols=125,unit='req')+'\n',method)
                            for method,cells in report['cells'].items():
                                for cell,value in cells.items():
                                    if value['complete'] and (method,cell) not in rendered:
                                        emit(method+' completed\n'+baseline_table({cell:value}),method);rendered.add((method,cell))
                        if jobs and not running and time.monotonic()-last_wait>=60:
                            emit('Waiting for an idle authorized GPU; other processes are left untouched.',current_method(jobs,running));last_wait=time.monotonic()
                    if len(rows)!=total:raise RuntimeError('Incomplete baseline stage')
                    report=status('complete' if phase=='full' else 'smoke_complete')
                    write_json(args.output/(phase+'_summary.json'),report)
                    for method,cells in report['cells'].items():emit(method+' '+phase+' results\n'+baseline_table(cells),method)
                    emit(phase+' complete; warm/timed token/text/NFE checks all passed')
                except BaseException:
                    status('failed')
                    for job,proc in running.values():
                        if proc.poll() is None:proc.send_signal(signal.SIGINT)
                    raise
                finally:bar.close()
            (args.output/'complete').write_text('OK\n');(args.output/'exit_code').write_text('0\n')
        except BaseException:
            (args.output/'exit_code').write_text('1\n')
            raise


def main():
    # Worker mode has no campaign arguments or filesystem side effects.
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        worker(sys.argv[2]);return
    campaign(parser().parse_args())


if __name__=='__main__':main()
