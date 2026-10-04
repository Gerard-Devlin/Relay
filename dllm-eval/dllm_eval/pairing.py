"""Historical fixed-128 screening aggregation, without competing generators."""
import statistics
from .evaluation import paired_intervals
TASKS=("gsm8k","humaneval","mbpp","math")
METHODS=("flash_cache","flash_verify","relay_cache","r1")
def summarize(rows):
    report={}
    for task in (*TASKS,'pooled'):
        cells={}
        for method in METHODS:
            part=[r for r in rows if r['method']==method and (task=='pooled' or r['task']==task)]
            if not part:continue
            cells[method]=dict(samples=len(part),correct=sum(r['correct'] for r in part),
                accuracy_percent=100*sum(r['correct'] for r in part)/len(part),
                mean_seconds=statistics.mean(r['seconds'] for r in part),mean_nfe=statistics.mean(r['nfe'] for r in part),
                mean_ordinary_calls=statistics.mean(r['ordinary_calls'] for r in part),
                mean_private_calls=statistics.mean(r['private_calls'] for r in part),
                mean_private_accepted=statistics.mean(r['private_accepted'] for r in part),
                mean_output_tokens=statistics.mean(r['output_tokens'] for r in part),truncated=sum(r['truncated'] for r in part),
                mean_actual_ordinary_row_layers=statistics.mean(r['actual_ordinary_row_layers'] for r in part),
                mean_optional_row_layers_skipped=statistics.mean(r['optional_row_layers_skipped'] for r in part),
                max_boundary_bytes=max(r['boundary_bytes'] for r in part))
        report[task]=cells
    return report

def finish_report(report):
    rows=report['rows'];pairs={(r['task'],r['id']) for r in rows}
    assert len(rows)==512 and len(pairs)==128
    assert all(sum(task==t for task,_ in pairs)==32 for t in TASKS)
    assert all(len({r['gpu'] for r in rows if r['task']==t and r['id']==i})==1 for t,i in pairs)
    assert all(sum(r['task']==t and r['id']==i and r['method']==m for r in rows)==1
               for t,i in pairs for m in METHODS)
    report['summary']=summarize(rows);paired={};discordant={}
    index={(r['task'],r['id'],r['method']):r for r in rows}
    for group in (*TASKS,'pooled'):
        selected=sorted((t,i) for t,i in pairs if group=='pooled' or group==t)
        paired[group]={};discordant[group]={}
        for baseline in ('flash_cache','flash_verify','r1'):
            a=[index[t,i,baseline] for t,i in selected]
            b=[index[t,i,'relay_cache'] for t,i in selected]
            paired[group][baseline]=paired_intervals([r['correct'] for r in a],[r['correct'] for r in b],
                [r['seconds'] for r in a],[r['seconds'] for r in b])
            discordant[group][baseline]=dict(
                baseline_only_correct=sum(x['correct'] and not y['correct'] for x,y in zip(a,b)),
                relay_only_correct=sum(y['correct'] and not x['correct'] for x,y in zip(a,b)),
                same_correctness=sum(x['correct']==y['correct'] for x,y in zip(a,b)))
    cells=report['summary'];pooled=cells['pooled']
    parts=dict(speed=paired['pooled']['flash_cache']['pooled_speedup']>=1.15,
        point_quality=pooled['relay_cache']['correct']>=pooled['flash_cache']['correct']-2,
        task_quality=all(cells[t]['relay_cache']['correct']>=cells[t]['flash_cache']['correct']-2 for t in TASKS),
        task_cost=all(cells[t]['relay_cache']['mean_seconds']<=1.10*cells[t]['flash_cache']['mean_seconds'] for t in TASKS))
    report.update(paired=paired,discordant=discordant,gates=dict(relay_cache=dict(parts=parts,
        continue_candidate=all(parts.values()))),goal_achieved=False,continuing_gate=all(parts.values()))
    return report
