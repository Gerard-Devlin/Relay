"""Exact six-way coverage of the frozen eight benchmark cells."""
import hashlib,json,random

# Default historical shard labels, not physical-device UUID bindings.
GPUS={i:i for i in range(2,8)}

COUNTS={'gsm8k':1319,'humaneval':164,'mbpp':500,'math':5000}
CELLS=tuple((task,length) for length in (256,512) for task in COUNTS)
SETTINGS=dict(method='relay_cache',stripe_width=8,history_budget=128,lengths=[256,512],
    block=32,threshold=.9,gamma=.8,track=4,mask=4,verify=False,seed=51713,precision='BF16',
    warmup='Every prompt has one excluded warm request immediately before timed replay',
    baseline_regenerated=False,algorithm_retuned=False)

def sample_id(sample):
    value=sample.get('id',sample.get('task_id'))
    if value is None:raise ValueError('Dataset ID missing')
    return str(value)

def shard(rows,gpu):
    if gpu not in GPUS:raise ValueError('Unauthorized physical GPU')
    ids=[sample_id(s) for s in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate dataset IDs')
    order=list(enumerate(rows));random.Random(SETTINGS['seed']).shuffle(order)
    return order[list(GPUS).index(gpu)::len(GPUS)]

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def cell_summary(rows,expected):
    if len(rows)>expected or len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate/excess cell rows')
    return dict(expected=expected,generated=len(rows),scored=len(rows),complete=len(rows)==expected,
        correct=sum(r['correct'] for r in rows),sum_seconds=sum(r['seconds'] for r in rows),
        sum_nfe=sum(r['nfe'] for r in rows),sum_ordinary_calls=sum(r['ordinary_calls'] for r in rows),
        sum_private_calls=sum(r['private_calls'] for r in rows),sum_truncated=sum(r['truncated'] for r in rows),
        sum_output_tokens=sum(r['output_tokens'] for r in rows),
        sum_actual_row_layers=sum(r['actual_ordinary_row_layers'] for r in rows),
        sum_optional_skipped_row_layers=sum(r['optional_row_layers_skipped'] for r in rows),
        max_boundary_bytes=max((r['boundary_bytes'] for r in rows),default=0))
