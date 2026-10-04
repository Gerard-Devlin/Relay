"""Evaluate official external LLaDA methods with Relay's unchanged prompts and scorer."""
import json
import os
import time
from pathlib import Path

from relay_cache.guards import (ROOT, verify_sources, verify_environment, fingerprint,
                                exclusive_lock, gpu_lease, check_binding, validate_resume, validate_settings)
from relay_cache.utils import write_json, sha256, generation_prompt
from .run import parser as base_parser, read_data, prepare_run
from .baseline import settings, source_manifest, verify_upstream, Session, assert_same_generation, batches, batch_metrics, baseline_table
from .evaluation import evaluate
from .scoring_guard import installed
from .score_answers import policy_hash
from .reporting import evaluation_log


class BaselineReporter:
    """Use the shared log transport without labelling batch throughput as request latency."""
    def __init__(self,base):self.base=base
    def info(self,message):self.base.info(message)
    def start_run(self,args,samples):
        self.info(f'{args.method} | GPU {args.gpu} | shard {args.rank+1}/{args.world_size}')
        self.info('Warmed batch latency and amortized time/item; setup, preparation, warm-up, scoring and I/O excluded.')
    def start_cell(self,task,length,total):
        self.base.start_cell(task,length,total);self.key=f'{task}_{length}';self.rows=[]
    def update(self,row):
        self.rows.append(row);cell=batch_metrics(self.rows)
        self.base.bar.set_postfix_str(f"acc={cell['accuracy_percent']:.2f}%, time/item={cell['mean_seconds']:.3f}s",refresh=False)
        self.base.bar.update(1)
    def finish_cell(self):
        self.base.close_progress();self.info('Completed '+self.key)
        print(baseline_table({self.key:batch_metrics(self.rows)}),flush=True)
    def finish(self,cells,output):
        self.info('Evaluation complete');print(baseline_table(cells),flush=True)
        self.info('Summary: '+str(output/'summary.json'))


def parser():
    p = base_parser()
    p.description = __doc__
    p.add_argument('--method', choices=('elastic_cache', 'd2cache'), required=True)
    p.add_argument('--baseline-source', type=Path, required=True)
    return p


def run_evaluation(args, reporter):
    if args.model != 'llada':
        raise ValueError('This baseline comparison is LLaDA only')
    args.config = args.config or ROOT / 'dllm-eval/configs/reproduction.json'
    config = json.loads(args.config.read_text())
    validate_settings(config['settings'])
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.lengths)) != len(args.lengths):
        raise ValueError('Duplicate task or length')
    sources = verify_sources()
    upstream = source_manifest(args.method, args.baseline_source)
    env = verify_environment()
    samples, data = read_data(args, config)
    for value, name in ((args.hf_home, 'HF_HOME'), (args.hf_hub_cache, 'HF_HUB_CACHE')):
        if value:
            os.environ[name] = str(value)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    profile = settings(args.method)
    manifest = dict(schema=1, sources=sources, upstream=upstream, environment=env, settings=profile,
                    datasets=data, config_sha256=sha256(args.config), lengths=args.lengths, tasks=args.tasks,
                    rank=args.rank, world_size=args.world_size, policy_sha256=policy_hash(),
                    model=profile['model'], revision=profile['revision'],
                    input_whitelist='paper_prompt or prompt only; references accessed after generation persistence')
    with exclusive_lock(args.output.parent / (args.output.name + '.lock')):
        prepare_run(args.output, manifest, args.resume)
        with gpu_lease(args.gpu) as gpu:
            write_json(args.output / 'resource.json', dict(gpu=gpu, binding=check_binding()))
            reporter.info('Official external method: ' + args.method + ' | ' + profile['profile'])
            reporter.start_run(args, samples)
            session = None
            scalars, events = [], []
            for length in args.lengths:
                for task in args.tasks:
                    folder = args.output / f'{task}_{length}' / 'records'
                    folder.mkdir(parents=True, exist_ok=True)
                    reporter.start_cell(task, length, len(samples[task]))
                    for group in batches(samples[task],profile['batch_size']):
                        identities=[str(s.get('id',s.get('task_id'))) for s in group]
                        prompts=[generation_prompt(s) for s in group]
                        prompt_hashes=[fingerprint(p) for p in prompts]
                        batch_id=fingerprint([args.method,task,length,identities])
                        batch_path=folder.parent/'batches'/(batch_id+'.json')
                        if batch_path.exists():
                            saved=json.loads(batch_path.read_text())
                            if (saved['ids'],saved['prompt_hashes'],saved['manifest_sha256'])!=(identities,prompt_hashes,sha256(args.output/'manifest.json')):
                                raise RuntimeError('Saved batch identity or manifest changed')
                        else:
                            if session is None:
                                began=time.perf_counter();session=Session(args.method,args.baseline_source)
                                write_json(args.output/'setup.json',dict(model_load_seconds=time.perf_counter()-began))
                            began=time.perf_counter();prompt=session.prepare_batch(prompts,task);prepared=time.perf_counter()-began
                            warm=session.generate_batch(prompt,length,group,task,batch_id)
                            results=session.generate_batch(prompt,length,group,task,batch_id)
                            for a,b in zip(warm,results):assert_same_generation(a,b)
                            saved=dict(ids=identities,prompt_hashes=prompt_hashes,results=results,
                                prepare_seconds_excluded=prepared,warm_batch_seconds_excluded=warm[0]['batch_seconds'],
                                manifest_sha256=sha256(args.output/'manifest.json'))
                            write_json(batch_path,saved)
                        if len(saved['results'])!=len(group):
                            raise RuntimeError('Saved batch result count changed')
                        for index,sample in enumerate(group):
                            record_sample(args,reporter,scalars,events,folder,saved,index,sample,task,length,upstream)
                    reporter.finish_cell()
            cells = {}
            for task in args.tasks:
                for length in args.lengths:
                    rows = [r for r in scalars if r['task']==task and r['length']==length]
                    if len(rows) != len(samples[task]):raise RuntimeError('Incomplete evaluation cell')
                    cells[f'{task}_{length}']=batch_metrics(rows)
            verify_sources()
            validate_resume(json.loads((args.output/'manifest.json').read_text()), manifest)
            write_json(args.output/'summary.json', dict(status='complete', method=args.method, cells=cells,
                scored=len(scalars), scoring_compatibility_events=events,
                scope='Warmed generation time; batch latency and amortized time per item are separate. Setup, prompt preparation, warm-up, postprocessing, scoring and I/O excluded'))
            (args.output/'complete').write_text('OK\n')
            reporter.finish(cells, args.output)


def record_sample(args,reporter,scalars,events,folder,saved,index,sample,task,length,upstream):
    ident=str(sample.get('id',sample.get('task_id')))
    path=folder/(fingerprint([task,length,ident])+'.json')
    prompt_hash=fingerprint(generation_prompt(sample))
    if path.exists():
        row=json.loads(path.read_text())
        if (row['task'],row['length'],row['id'],row['prompt_sha256'])!=(task,length,ident,prompt_hash):
            raise RuntimeError('Saved prompt identity changed')
        if row['result']!=saved['results'][index]:
            raise RuntimeError('Saved record differs from its persisted batch')
    else:
        row=dict(task=task,length=length,id=ident,prompt_sha256=prompt_hash,result=saved['results'][index],
            prepare_seconds_excluded=saved['prepare_seconds_excluded']/len(saved['ids']),
            warm_seconds_excluded=saved['warm_batch_seconds_excluded']/len(saved['ids']),
            manifest_sha256=sha256(args.output/'manifest.json'))
        write_json(path,row)
    if row['manifest_sha256']!=sha256(args.output/'manifest.json'):
        raise RuntimeError('Saved generation manifest changed')
    if 'assessment' not in row:
        with installed(events):row['assessment']=evaluate(row['result']['text'],sample,task)
        write_json(path,row)
    result=row['result']
    scalar=dict(task=task,length=length,id=ident,correct=bool(row['assessment']['correct']),
        seconds=result['seconds'],nfe=result['nfe'],method=args.method,
        batch_id=result['batch_id'],batch_seconds=result['batch_seconds'],batch_size=result['batch_size'])
    scalars.append(scalar)
    verify_sources();verify_upstream(args.baseline_source,upstream)
    reporter.update(scalar)


def main():
    args = parser().parse_args()
    with evaluation_log(args.log_dir, args.output, args.rank) as reporter:
        run_evaluation(args, BaselineReporter(reporter))


if __name__ == '__main__':
    main()
