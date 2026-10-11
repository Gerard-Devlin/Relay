"""Reproduce the historical LLaDA-V profile; persist generations before scoring."""
import argparse,json,os,time
from pathlib import Path
from relay_cache.guards import ROOT,fingerprint,verify_sources,verify_environment,validate_resume
from relay_cache.utils import write_json,sha256
from relay_cache.lladav.settings import SETTINGS,validate_settings
from .reporting import evaluation_log,result_table

COUNTS={'mathvista':1000,'mathverse':2364}

def image_path(root,value):
    root=Path(root).resolve();path=(root/value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Prepared image path must stay inside data root')
    return path

def read_data(root):
    data=json.loads((root/'data.json').read_text(encoding='utf-8'))
    if set(data)!=set(COUNTS):raise ValueError('Expected both fixed visual datasets')
    for task,rows in data.items():
        if len(rows)!=COUNTS[task] or len({str(r['id']) for r in rows})!=len(rows):
            raise ValueError('Visual dataset count or unique IDs changed')
        for row in rows:
            if row['task']!=task or not isinstance(row['prompt'],str) or not row['prompt']:
                raise ValueError('Invalid visual model input')
            image_path(root,row['image'])
    return data

def validate_record(row,sample,identity):
    if row['identity']!=identity or row['id']!=sample['id'] or row['prompt_sha256']!=fingerprint(sample['prompt']) or row['image_sha256']!=sample['image_sha256']:
        raise RuntimeError('Saved visual record does not match the current run')
    result=row['result']
    if not isinstance(result['token_ids'],list) or not result['token_ids'] or result['nfe']<=0:
        raise RuntimeError('Incomplete saved generation')

def run(args,reporter):
    from relay_cache.guards import gpu_lease,check_binding,exclusive_lock
    from relay_cache.lladav.environment import verify_visual_environment
    config=json.loads(args.config.read_text());validate_settings(config['settings'])
    if len(set(args.tasks))!=len(args.tasks) or len(set(args.lengths))!=len(args.lengths):raise ValueError('Duplicate cell')
    if args.limit is not None and args.limit<=0:raise ValueError('Limit must be positive')
    if not 0<=args.rank<args.world_size:raise ValueError('Invalid shard')
    sources=verify_sources();env=verify_environment();vision_env=verify_visual_environment();data=read_data(args.data_root)
    manifest=dict(schema=1,settings=SETTINGS,sources=sources,environment=env,vision_environment=vision_env,
        data_sha256=sha256(args.data_root/'data.json'),config_sha256=sha256(args.config),
        tasks=args.tasks,lengths=args.lengths,limit=args.limit,rank=args.rank,world_size=args.world_size,
        measurement=config['measurement'],model_inputs='prompt and RGB image only; references not passed to generation')
    identity=fingerprint(manifest)
    os.environ['HF_HUB_CACHE']=str(args.hf_hub_cache)
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    with exclusive_lock(args.output.parent/(args.output.name+'.lock')):
        if args.output.exists():
            if not args.resume:raise FileExistsError('Output exists; use --resume explicitly')
            validate_resume(json.loads((args.output/'manifest.json').read_text()),manifest)
        else:
            if args.resume:raise FileNotFoundError('Cannot resume an absent run')
            args.output.mkdir(parents=True);write_json(args.output/'manifest.json',manifest)
        with gpu_lease(args.gpu) as gpu:
            check_binding()
            import torch
            from PIL import Image
            from tqdm import tqdm
            from relay_cache.lladav.generate import Session
            from .visual_output import for_scoring
            start=time.perf_counter();session=Session(args.hf_hub_cache)
            write_json(args.output/'loading.json',dict(gpu=gpu,loading=session.loading,mapping=session.mapping,seconds=time.perf_counter()-start))
            first=data['mathvista'][0]
            with Image.open(image_path(args.data_root,first['image'])) as image:warm=session.prepare(first['prompt'],image)
            for _ in range(2):session.generate(warm,256)
            del warm
            reporter.info('Two startup warmups complete. Formal requests run once; scoring is a separate saved-output pass.')
            cells={}
            for task in args.tasks:
                samples=data[task][:args.limit] if args.limit is not None else data[task]
                indexed=list(enumerate(samples))[args.rank::args.world_size]
                for length in args.lengths:
                    rows=[]
                    with tqdm(total=len(indexed),desc=f'{task} / {length}',file=reporter.progress_stream,ascii=False,
                        ncols=110,mininterval=1,bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as bar:
                        for index,sample in indexed:
                            path=args.output/'records'/task/str(length)/f'{index:05d}.json'
                            if path.exists():
                                row=json.loads(path.read_text());validate_record(row,sample,identity)
                            else:
                                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                                with Image.open(image_path(args.data_root,sample['image'])) as image:value=session.prepare(sample['prompt'],image)
                                torch.cuda.synchronize();prepare_seconds=time.perf_counter()-start
                                if value['image_sha256']!=sample['image_sha256']:raise RuntimeError('Image pixels changed')
                                result=session.generate(value,length)
                                scored_text,completion=for_scoring(result,session.tokenizer,value['stop'],length);del value
                                row=dict(identity=identity,task=task,length=length,index=index,id=sample['id'],
                                    prompt_sha256=fingerprint(sample['prompt']),image_sha256=sample['image_sha256'],
                                    result=result,scored_text=scored_text,completion=completion,prepare_seconds=prepare_seconds,
                                    peak_memory=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved()))
                                write_json(path,row)
                            rows.append(row);bar.update()
                    n=len(rows)
                    if n:
                        cells[f'{task}_{length}']=dict(samples=n,correct='pending',accuracy_percent=None,
                            mean_seconds=sum(r['result']['seconds'] for r in rows)/n,
                            mean_prepare_seconds=sum(r['prepare_seconds'] for r in rows)/n,
                            mean_nfe=sum(r['result']['nfe'] for r in rows)/n,
                            peak_allocated_bytes=max(r['peak_memory']['allocated_bytes'] for r in rows),
                            peak_reserved_bytes=max(r['peak_memory']['reserved_bytes'] for r in rows))
                        print(result_table({f'{task}_{length}':cells[f'{task}_{length}']}),flush=True)
            verify_sources();write_json(args.output/'summary.json',dict(status='generated',cells=cells,identity=identity))
            reporter.info('Generations saved. Run dllm_eval.visual_score to calculate accuracy without running the model again.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'dllm-eval/configs/lladav.json')
    p.add_argument('--data-root',required=True,type=Path);p.add_argument('--hf-hub-cache',required=True,type=Path)
    p.add_argument('--gpu',required=True);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--tasks',nargs='+',choices=tuple(COUNTS),default=list(COUNTS))
    p.add_argument('--lengths',nargs='+',type=int,choices=(256,512),default=[256,512])
    p.add_argument('--limit',type=int);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--resume',action='store_true');p.add_argument('--log-dir',type=Path,default=ROOT/'log')
    args=p.parse_args()
    with evaluation_log(args.log_dir,args.output,args.rank) as reporter:run(args,reporter)

if __name__=='__main__':main()
