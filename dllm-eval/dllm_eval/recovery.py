"""Pure disjoint recovery plan; existing generation must never be regenerated."""
from collections import Counter
from .plan import COUNTS,CELLS,GPUS,sample_id

PHYSICAL=(2,3,4,5,6)


def split_missing(datasets, existing_keys):
    existing=set(existing_keys)
    if len(existing)!=len(existing_keys):
        raise ValueError('Duplicate original request identities')
    expected={(task,length,sample_id(row)) for task,length in CELLS for row in datasets[task]}
    if not existing<=expected:
        raise ValueError('Original requests outside the fixed benchmark')
    assignments={str(length):{str(g):{} for g in PHYSICAL} for length in (256,512)}
    for task,length in CELLS:
        for row in datasets[task]:
            if len(datasets[task])!=COUNTS[task]:
                raise ValueError('Fixed dataset coverage changed')
        # Original full dataset order is deterministic; all missing requests
        # remain the original (task,length,id), irrespective of new physical GPU.
        missing=[dict(index=i,id=sample_id(row)) for i,row in enumerate(datasets[task])
            if (task,length,sample_id(row)) not in existing]
        for g in PHYSICAL:
            assignments[str(length)][str(g)][task]=missing[PHYSICAL.index(g)::len(PHYSICAL)]
    recovered=[(task,int(length),row['id']) for length,workers in assignments.items()
        for tasks in workers.values() for task,rows in tasks.items() for row in rows]
    if len(recovered)!=len(set(recovered)) or set(recovered)!=expected-existing:
        raise AssertionError('Recovery does not exactly cover missing requests')
    return assignments
