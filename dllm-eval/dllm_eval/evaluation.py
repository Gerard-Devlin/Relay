"""Unchanged grades; references accessed only after generation is persisted."""
def evaluate(text,sample,task):
    if task in ('gsm8k','math'):
        from dllm_eval.score_answers import assess
        if task=='gsm8k':gold=sample['answer'].rsplit('####',1)[-1].strip()
        else:
            from dllm_eval.score_math import installed_utils,load_metric
            metric=load_metric(installed_utils());gold=metric['remove_boxed'](metric['last_boxed_only_string'](sample['solution']))
        return assess(text,gold,task)
    if task=='humaneval':
        from dllm_eval.score_humaneval import clean_completion,check
        code=clean_completion(sample['prompt'],text,sample['entry_point'])
        return dict(correct=check(code,sample['test'],sample['entry_point'],6),policy='official execution')
    from dllm_eval.score_mbpp import clean_completion,check
    return dict(correct=check(clean_completion(text),sample['test_list'],6),policy='official execution')

def paired_intervals(before,after,seconds_before,seconds_after):
    import numpy as np
    n=len(before)
    if not n or any(len(v)!=n for v in (after,seconds_before,seconds_after)):
        raise ValueError('Complete paired records required')
    if min([*seconds_before,*seconds_after])<=0:raise ValueError('Positive full-request timing required')
    rng=np.random.default_rng(1234);indices=rng.integers(0,n,(10000,n))
    delta=np.asarray(after,dtype=float)-np.asarray(before,dtype=float)
    speed=np.asarray(seconds_before)[indices].sum(1)/np.asarray(seconds_after)[indices].sum(1)
    return dict(accuracy_difference_pp=float(delta.mean()*100),
                accuracy_difference_95ci_pp=(np.percentile(delta[indices].mean(1),[2.5,97.5])*100).tolist(),
                pooled_speedup=sum(seconds_before)/sum(seconds_after),
                speedup_95ci=np.percentile(speed,[2.5,97.5]).tolist(),bootstrap_samples=10000,seed=1234,
                scope='Paired screening interval; does not establish statistical losslessness')
