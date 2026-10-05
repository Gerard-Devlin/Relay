"""Two startup requests per fresh model/GPU process, excluded from metrics."""
import os,time
from relay_cache.utils import generation_prompt,write_json
from relay_cache.guards import fingerprint

DESCRIPTION='Two excluded startup requests per algorithm worker; each measured request generated once'
POLICY=dict(name='startup_two_requests_v1',requests_per_worker=2,scope='fresh model process / physical GPU',
    length='first requested generation length',per_prompt_replay=False,excluded_from_metrics=True)


def requests(samples,tasks):
    chosen=[]
    for task in tasks:
        for sample in samples[task]:
            generation_prompt(sample)
            chosen.append((task,sample))
            if len(chosen)==2:return chosen
    # A single-request smoke still executes exactly two startup requests.
    return chosen*2 if len(chosen)==1 else chosen


def startup(session,samples,args,reporter,method=None):
    selected=requests(samples,args.tasks)
    if not selected:raise ValueError('Startup warm-up requires a selected prompt')
    length=args.lengths[0];records=[];began=time.perf_counter()
    reporter.info('Startup warm-up: two requests on this worker; excluded from time/accuracy/NFE metrics')
    for index,(task,sample) in enumerate(selected):
        text=generation_prompt(sample)
        if method is None:
            prompt=session.prepare(text,task)
            result=session.generate(prompt,length,sample,task)
        else:
            prompt=session.prepare_batch([text],task)
            result=session.generate_batch(prompt,length,[sample],task,f'warmup-{index}')[0]
        records.append(dict(task=task,id=str(sample.get('id',sample.get('task_id'))),length=length,
            prompt_sha256=fingerprint(text),seconds=result['seconds'],nfe=result['nfe']))
    write_json(args.output/f'warmup_{os.getpid()}_{time.time_ns()}.json',
        dict(policy=POLICY,method=method or 'relay',requests=records,wall_seconds=time.perf_counter()-began))
    reporter.info('Startup warm-up complete; every subsequent request is generated once')
