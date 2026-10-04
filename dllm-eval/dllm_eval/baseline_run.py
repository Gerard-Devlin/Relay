"""Evaluate official external LLaDA methods with Relay's unchanged prompts and scorer."""
import json
import os
import time
from pathlib import Path

from relay_cache.guards import (ROOT, verify_sources, verify_environment, fingerprint,
                                exclusive_lock, gpu_lease, check_binding, validate_resume, validate_settings)
from relay_cache.utils import write_json, sha256, generation_prompt
from .run import parser as base_parser, read_data, prepare_run
from .baseline import settings, source_manifest, verify_upstream, Session, assert_same_generation
from .evaluation import evaluate
from .scoring_guard import installed
from .score_answers import policy_hash
from .reporting import evaluation_log


def parser():
    p = base_parser()
    p.description = __doc__
    p.add_argument('--method', choices=('es_dllm', 'sparsed'), required=True)
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
                    for sample in samples[task]:
                        ident = str(sample.get('id', sample.get('task_id')))
                        path = folder / (fingerprint([task, length, ident]) + '.json')
                        prompt_hash = fingerprint(generation_prompt(sample))
                        if path.exists():
                            row = json.loads(path.read_text())
                            if (row['task'], row['length'], row['id'], row['prompt_sha256']) != (task, length, ident, prompt_hash):
                                raise RuntimeError('Saved prompt identity changed')
                        else:
                            if session is None:
                                began = time.perf_counter()
                                session = Session(args.method, args.baseline_source)
                                write_json(args.output/'setup.json', dict(model_load_seconds=time.perf_counter()-began))
                            began = time.perf_counter()
                            prompt = session.prepare(generation_prompt(sample), task)
                            prepared = time.perf_counter()-began
                            warm = session.generate(prompt, length, sample, task)
                            result = session.generate(prompt, length, sample, task)
                            assert_same_generation(warm, result)
                            row = dict(task=task, length=length, id=ident, prompt_sha256=prompt_hash, result=result,
                                       prepare_seconds_excluded=prepared, warm_seconds_excluded=warm['seconds'],
                                       manifest_sha256=sha256(args.output/'manifest.json'))
                            write_json(path, row)
                        if row['manifest_sha256'] != sha256(args.output/'manifest.json'):
                            raise RuntimeError('Saved generation manifest changed')
                        if 'assessment' not in row:
                            with installed(events):
                                row['assessment'] = evaluate(row['result']['text'], sample, task)
                            write_json(path, row)
                        result = row['result']
                        scalar = dict(task=task, length=length, id=ident, correct=bool(row['assessment']['correct']),
                                      seconds=result['seconds'], nfe=result['nfe'], method=args.method)
                        scalars.append(scalar)
                        verify_sources()
                        verify_upstream(args.baseline_source, upstream)
                        reporter.update(scalar)
                    reporter.finish_cell()
            cells = {}
            for task in args.tasks:
                for length in args.lengths:
                    rows = [r for r in scalars if r['task']==task and r['length']==length]
                    n = len(rows)
                    if n != len(samples[task]):
                        raise RuntimeError('Incomplete evaluation cell')
                    correct = sum(r['correct'] for r in rows)
                    cells[f'{task}_{length}'] = dict(samples=n, correct=correct,
                        accuracy_percent=100*correct/n if n else None,
                        mean_seconds=sum(r['seconds'] for r in rows)/n if n else None,
                        mean_nfe=sum(r['nfe'] for r in rows)/n if n else None)
            verify_sources()
            validate_resume(json.loads((args.output/'manifest.json').read_text()), manifest)
            write_json(args.output/'summary.json', dict(status='complete', method=args.method, cells=cells,
                scored=len(scalars), scoring_compatibility_events=events,
                scope='Warmed prepared-prompt requests; model loading, prompt preparation, warm-up, postprocessing, scoring and I/O excluded'))
            (args.output/'complete').write_text('OK\n')
            reporter.finish(cells, args.output)


def main():
    args = parser().parse_args()
    with evaluation_log(args.log_dir, args.output, args.rank) as reporter:
        run_evaluation(args, reporter)


if __name__ == '__main__':
    main()
