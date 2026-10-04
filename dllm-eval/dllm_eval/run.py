"""Durable serial Relay evaluation, with explicit GPU leasing and exact resume."""
import argparse,json,os,time
from pathlib import Path
from relay_cache.utils import sha256,write_json
from relay_cache.guards import SETTINGS,validate_settings
from relay_cache.utils import generation_prompt,select_samples
from relay_cache.guards import ROOT,verify_sources,verify_environment,fingerprint,validate_resume
from .reporting import evaluation_log
from .plan import COUNTS

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'dllm-eval/configs/reproduction.json')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dataset',action='append',default=[],metavar='TASK=PATH')
    p.add_argument('--gpu',required=True,help='Idle physical index or UUID')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--log-dir',type=Path,default=ROOT/'log',help='One transcript per invocation (default: repository/log)')
    p.add_argument('--hf-home',type=Path);p.add_argument('--hf-hub-cache',type=Path)
    p.add_argument('--tasks',nargs='+',choices=('gsm8k','humaneval','mbpp','math'),default=list(('gsm8k','humaneval','mbpp','math')))
    p.add_argument('--lengths',nargs='+',type=int,choices=(256,512),default=[256,512])
    p.add_argument('--limit',type=int);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    return p

def read_data(args,config):
    expected=COUNTS;paths=dict(config['datasets'])
    for entry in args.dataset:
        task,path=entry.split('=',1)
        if task not in expected:raise ValueError('Unknown dataset override')
        paths[task]=path
    samples={};provenance={}
    if not 0<=args.rank<args.world_size:raise ValueError('Invalid shard')
    for task in args.tasks:
        path=(args.data_root/paths[task]).resolve()
        rows=json.loads(path.read_text(encoding='utf-8'))
        if len(rows)!=expected[task]:raise RuntimeError('Fixed dataset count mismatch')
        chosen=select_samples(path,args.limit or len(rows),args.offset,SETTINGS['seed'])
        for sample in chosen:generation_prompt(sample)
        samples[task]=chosen[args.rank::args.world_size]
        provenance[task]=dict(sha256=sha256(path),count=len(rows),selected_ids=[str(s.get('id',s.get('task_id'))) for s in chosen])
    return samples,provenance

def prepare_run(output,manifest,resume=False):
    if output.exists():
        if not resume:raise FileExistsError('Output exists; explicit --resume required')
        validate_resume(json.loads((output/'manifest.json').read_text()),manifest)
    else:
        if resume:raise FileNotFoundError('Cannot resume absent run')
        output.mkdir(parents=True);write_json(output/'manifest.json',manifest)

def run_evaluation(args,reporter):
    reporter.info('Validating fixed sources, configuration, environment and datasets')
    config=json.loads(args.config.read_text());validate_settings(config['settings'])
    if len(set(args.tasks))!=len(args.tasks) or len(set(args.lengths))!=len(args.lengths):raise ValueError('Duplicate task or length')
    sources=verify_sources();env=verify_environment();samples,data=read_data(args,config)
    for value,name in ((args.hf_home,'HF_HOME'),(args.hf_hub_cache,'HF_HUB_CACHE')):
        if value:os.environ[name]=str(value)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from relay_cache.guards import gpu_lease,exclusive_lock,check_binding
    from .evaluation import evaluate
    from .scoring_guard import installed
    from .score_answers import policy_hash
    manifest=dict(schema=1,sources=sources,environment=env,settings=SETTINGS,datasets=data,config_sha256=sha256(args.config),
        lengths=args.lengths,tasks=args.tasks,rank=args.rank,world_size=args.world_size,policy_sha256=policy_hash(),
        model='GSAI-ML/LLaDA-8B-Instruct',revision='08b83a6feb34df1a6011b80c3c00c7563e963b07',
        input_whitelist='paper_prompt or prompt only; references are accessed after generation persistence')
    with exclusive_lock(args.output.parent/(args.output.name+'.lock')):
        prepare_run(args.output,manifest,args.resume)
        with gpu_lease(args.gpu) as gpu:
            binding=check_binding();write_json(args.output/'resource.json',dict(gpu=gpu,binding=binding))
            reporter.start_run(args,samples)
            from relay_cache.generate import Session,assert_same_generation
            session=None;events=[];all_scalars=[]
            for length in args.lengths:
                for task in args.tasks:
                    folder=args.output/f'{task}_{length}'/'records';folder.mkdir(parents=True,exist_ok=True)
                    reporter.start_cell(task,length,len(samples[task]))
                    for sample in samples[task]:
                        ident=str(sample.get('id',sample.get('task_id')));key=fingerprint([task,length,ident])
                        path=folder/(key+'.json');prompt_hash=fingerprint(generation_prompt(sample))
                        if path.exists():
                            row=json.loads(path.read_text())
                            if (row['task'],row['length'],row['id'],row['prompt_sha256'])!=(task,length,ident,prompt_hash):
                                raise RuntimeError('Existing request identity changed')
                            result=row['result']
                        else:
                            if session is None:
                                began=time.perf_counter();session=Session()
                                write_json(args.output/'setup.json',dict(model_load_seconds=time.perf_counter()-began,excluded_from_request=True))
                            began=time.perf_counter();prompt=session.prepare(generation_prompt(sample),task)
                            prepared=time.perf_counter()-began
                            warm=session.generate(prompt,length,sample,task);result=session.generate(prompt,length,sample,task)
                            assert_same_generation(warm,result)
                            row=dict(task=task,length=length,id=ident,prompt_sha256=prompt_hash,result=result,
                                prepare_seconds_excluded=prepared,warm_seconds_excluded=warm['seconds'],
                                manifest_sha256=sha256(args.output/'manifest.json'))
                            write_json(path,row)
                        if row.get('manifest_sha256')!=sha256(args.output/'manifest.json'):
                            raise RuntimeError('Saved generation belongs to another manifest')
                        if 'assessment' not in row:
                            with installed(events):row['assessment']=evaluate(result['text'],sample,task)
                            write_json(path,row)
                        all_scalars.append(dict(task=task,length=length,id=ident,correct=bool(row['assessment']['correct']),
                            seconds=result['seconds'],nfe=result['nfe']))
                        verify_sources();reporter.update(all_scalars[-1])
                    reporter.finish_cell()
            cells={}
            for task in args.tasks:
                for length in args.lengths:
                    part=[r for r in all_scalars if r['task']==task and r['length']==length]
                    if len(part)!=len(samples[task]):raise AssertionError('Incomplete cell')
                    n=len(part);cells[f'{task}_{length}']=dict(samples=n,correct=sum(r['correct'] for r in part),
                        accuracy_percent=100*sum(r['correct'] for r in part)/n if n else None,
                        mean_seconds=sum(r['seconds'] for r in part)/n if n else None,
                        mean_nfe=sum(r['nfe'] for r in part)/n if n else None)
            verify_sources();validate_resume(json.loads((args.output/'manifest.json').read_text()),manifest)
            write_json(args.output/'summary.json',dict(status='complete',cells=cells,scored=len(all_scalars),
                scoring_compatibility_events=events,scope='Warmed prepared-prompt requests; warm-up, setup, preparation, postprocessing, disk and scoring excluded'))
            (args.output/'complete').write_text('OK\n')
            reporter.finish(cells,args.output)

def main():
    args=parser().parse_args()
    with evaluation_log(args.log_dir,args.output,args.rank) as reporter:
        run_evaluation(args,reporter)

if __name__=='__main__':main()
