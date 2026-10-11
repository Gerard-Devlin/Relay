"""Generate from one question and image using the fixed historical profile."""
import argparse,json,os,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True,type=Path)
    p.add_argument('--prompt',required=True)
    p.add_argument('--length',type=int,choices=(256,512),default=256)
    p.add_argument('--gpu',required=True)
    p.add_argument('--hf-hub-cache',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    from relay_cache.guards import gpu_lease,check_binding,verify_sources,verify_environment,exclusive_lock
    from relay_cache.utils import write_json
    from .environment import verify_visual_environment
    sources=verify_sources();environment=verify_environment();vision_environment=verify_visual_environment()
    os.environ['HF_HUB_CACHE']=str(args.hf_hub_cache)
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    with exclusive_lock(args.output.with_suffix(args.output.suffix+'.lock')):
        if args.output.exists():raise FileExistsError('Output exists; choose a new output file')
        with gpu_lease(args.gpu) as gpu:
            check_binding()
            from PIL import Image
            from .generate import Session
            session=Session(args.hf_hub_cache)
            with Image.open(args.image) as image:value=session.prepare(args.prompt,image)
            for _ in range(2):session.generate(value,args.length)
            result=session.generate(value,args.length)
            from dllm_eval.visual_output import for_scoring
            scored_text,completion=for_scoring(result,session.tokenizer,value['stop'],args.length)
            verify_sources()
            write_json(args.output,dict(result=result,scored_text=scored_text,completion=completion,
                gpu=gpu,sources=sources,environment=environment,vision_environment=vision_environment,
                warmup='two excluded startup requests; one formal generation'))
            print(scored_text)

if __name__=='__main__':main()
