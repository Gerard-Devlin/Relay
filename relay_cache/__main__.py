"""Single generation with the frozen Relay configuration."""
import argparse,json,os
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__)
    q=p.add_mutually_exclusive_group(required=True);q.add_argument('--prompt');q.add_argument('--prompt-file',type=Path)
    p.add_argument('--gpu',required=True,help='Idle physical GPU index or UUID')
    p.add_argument('--length',type=int,choices=(256,512),default=256)
    p.add_argument('--task',choices=('gsm8k','math','humaneval','mbpp'),default='gsm8k')
    p.add_argument('--hf-home',type=Path);p.add_argument('--hf-hub-cache',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preformatted',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite generation')
    for value,name in ((args.hf_home,'HF_HOME'),(args.hf_hub_cache,'HF_HUB_CACHE')):
        if value:os.environ[name]=str(value)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from .guards import gpu_lease,check_binding
    from .guards import verify_sources,verify_environment
    from .utils import write_json
    sources=verify_sources();env=verify_environment()
    with gpu_lease(args.gpu) as gpu:
        binding=check_binding()
        from .generate import Session,assert_same_generation
        session=Session();text=args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding='utf-8')
        prompt=session.prepare(text,args.task,args.preformatted)
        warm=session.generate(prompt,args.length,{},args.task)
        result=session.generate(prompt,args.length,{},args.task);assert_same_generation(warm,result)
        verify_sources();write_json(args.output,dict(manifest=dict(sources=sources,environment=env,gpu=gpu,binding=binding),
            result=result,warm_seconds_excluded=warm['seconds']))
        print(result['text'])
if __name__=='__main__':main()
