"""Evaluation adapters for unmodified, pinned Elastic-Cache and d2Cache sources."""
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from relay_cache.utils import sha256, prompt_ids
from relay_cache.llada.loading import MODEL_ID, REVISION, snapshot
from relay_cache.llada.generate import postprocess_output

SOURCES = {
    'elastic_cache': ('https://github.com/VILA-Lab/Elastic-Cache.git', '1960d8fc6231205a1ae4ebba3898d475e339f7e1'),
    'd2cache': ('https://github.com/Kamichanw/d2Cache.git', '216b4557f4baf318a246763af805f414cbfb21a4'),
}
DREAM_SOURCES = {
    'fast_dllm_v1': ('https://github.com/NVlabs/Fast-dLLM.git', 'a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b'),
}
for _name in ('fast_dllm_v1_flash', 'fast_dllm_v1_no_flash'):
    DREAM_SOURCES[_name] = DREAM_SOURCES['fast_dllm_v1']
SOURCE_SUFFIXES = {'.py', '.json', '.yaml', '.yml'}


def official_token_environment():
    # Identical to d2Cache src.utils.pre_initialize for configs/model/llada-inst.yaml.
    for name, value in {'MASK_TOKEN_ID':126336, 'EOS_TOKEN_ID':126081, 'PAD_TOKEN_ID':126081}.items():
        os.environ[name] = str(value)


def settings(method):
    result = dict(method=method, model=MODEL_ID, revision=REVISION, precision='BF16',
                  seed=51713, lengths=[256, 512], sampling='official deterministic maskgit / low_confidence',
                  warmup='One excluded full request per prompt before timed replay', batch_size=1,
                  stop_until_eos=False, parallel_decoding=False)
    if method == 'elastic_cache':
        result.update(profile='Unmodified official Elastic-Cache LLaDA task scripts; checkpoint fixed for comparison',
                      window_length=16, threshold=.90, gamma=.90, track_num=1, block_caching=True,
                      tokens_per_iter=1, parallel_decoding=True, stop_until_eos=True,
                      backend='Official attention implementation with attention-weight tracking',
                      upstream_script_model='GSAI-ML/LLaDA-1.5; this comparison keeps the historical 8B-Instruct revision')
    elif method == 'd2cache':
        result.update(profile='Official semi-AR parallel example and paper Appendix E.2',
                      rollout_p=.1, current_k=32, sigma=10., inflate_w=4,
                      block=32, num_transfer_tokens=1, alg='maskgit_plus',
                      generation_sigma=0., threshold=.90, parallel_decoding=True,
                      backend='Official eager attention with attention rollout',
                      cli_dependencies=dict(omegaconf='2.3.0', hydra_core='1.3.2', antlr='4.9.3'))
    else:
        raise ValueError('Unknown external method')
    return result


def left_pad(token_rows, pad_id):
    if not token_rows or any(not row for row in token_rows):
        raise ValueError('Nonempty prompts required')
    width=max(map(len,token_rows))
    return ([([pad_id]*(width-len(row))+list(row)) for row in token_rows],
            [([0]*(width-len(row))+[1]*len(row)) for row in token_rows])


def files(source):
    return {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*')
            if p.is_file() and p.suffix in SOURCE_SUFFIXES
            and '.git' not in p.parts and '__pycache__' not in p.parts}


def source_manifest(method, source):
    source = Path(source).resolve()
    registry = {**SOURCES, **DREAM_SOURCES}
    # GitHub's pinned archive is usable when the server cannot reach Git transport.
    # Its receipt is generated at download time, outside the public package.
    receipt = source / '.official_archive.json'
    if method in DREAM_SOURCES and receipt.exists():
        declared = json.loads(receipt.read_text())
        url, revision = registry[method]
        if declared['revision'] != revision or declared['url'] != url:
            raise RuntimeError('Official archive revision mismatch')
        current = files(source)
        current.pop('.official_archive.json', None)
        if current != declared['files']:
            raise RuntimeError('Official archive files changed')
        return dict(url=url, revision=revision, files=files(source))
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    url, revision = registry[method]
    if actual != revision:
        raise RuntimeError('Upstream revision mismatch')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Upstream tracked files modified')
    return dict(url=url, revision=revision, files=files(source))


def verify_upstream(source, manifest):
    current = files(Path(source))
    if set(current) != set(manifest['files']):
        raise RuntimeError('Upstream source file set changed')
    for name, digest in manifest['files'].items():
        if current[name] != digest:
            raise RuntimeError('Upstream source changed: ' + name)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    try:
        spec.loader.exec_module(result)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return result


@contextmanager
def official_imports(source, dependencies=None):
    """Load optional CLI dependencies without replacing the scorer's ANTLR runtime."""
    old_path = sys.path[:]
    saved = {k: v for k, v in sys.modules.items() if k == 'antlr4' or k.startswith('antlr4.')}
    try:
        if dependencies is not None:
            for key in saved:
                sys.modules.pop(key)
            sys.path.insert(0, str(dependencies))
        sys.path.insert(0, str(source))
        yield
    finally:
        if dependencies is not None:
            for key in list(sys.modules):
                if key == 'antlr4' or key.startswith('antlr4.'):
                    sys.modules.pop(key)
            sys.modules.update(saved)
        sys.path[:] = old_path


@contextmanager
def elastic_transfers(generator):
    """Observe only the official sampler's returned indices; always restore its function."""
    namespace=inspect.unwrap(generator).__globals__
    original=namespace['get_decoded_token_confident'];counts=[]
    def observe(*args,**kwargs):
        result=original(*args,**kwargs)
        counts.append(result[1].numel())
        return result
    namespace['get_decoded_token_confident']=observe
    try:yield counts
    finally:namespace['get_decoded_token_confident']=original


class Session:
    def __init__(self, method, source):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.method, self.source = method, Path(source).resolve()
        self.settings = settings(method)
        self.upstream = source_manifest(method, self.source)
        torch.set_num_threads(1)
        torch.manual_seed(51713)
        checkpoint = snapshot()
        if method == 'elastic_cache':
            if 'model' in sys.modules:
                raise RuntimeError('Conflicting upstream model module namespace')
            with official_imports(self.source / 'llada'):
                from model.modeling_llada import LLaDAModelLM
                self.generator = module('_official_elastic_generate', self.source / 'llada/generate.py').generate_with_elastic_cache
            self.model, info = LLaDAModelLM.from_pretrained(checkpoint, local_files_only=True,
                    torch_dtype=torch.bfloat16, output_loading_info=True)
        else:
            if 'src' in sys.modules:
                raise RuntimeError('Conflicting upstream module namespace')
            official_token_environment()
            dependencies = self.source.parent.parent / 'deps'
            with official_imports(self.source, dependencies if dependencies.exists() else None):
                from src.cache import d2Cache
                from src.models.llada import LLaDAModelLM
                from src.generation import generate
                self.generator = generate
                self.cache_cls = partial(d2Cache, rollout_p=.1, current_k=32, sigma=10., inflate_w=4)
            self.model, info = LLaDAModelLM.from_pretrained(checkpoint, local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation='eager', output_loading_info=True)
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError('Original checkpoint does not match official model class: ' + repr(info))
        self.model = self.model.to('cuda:0').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

    def prepare(self, text, task):
        import torch
        return torch.tensor(prompt_ids(self.tokenizer, text, task, preformatted=True),
                            device='cuda:0', dtype=torch.long).unsqueeze(0)

    def prepare_batch(self, texts, task):
        import torch
        if not 1<=len(texts)<=self.settings['batch_size']:
            raise ValueError('Batch does not match the fixed official profile')
        rows=[prompt_ids(self.tokenizer,text,task,preformatted=True) for text in texts]
        pad_id=self.tokenizer.pad_token_id
        if pad_id is None:pad_id=self.tokenizer.eos_token_id
        ids,mask=left_pad(rows,pad_id)
        return dict(input_ids=torch.tensor(ids,device='cuda:0',dtype=torch.long),
                    attention_mask=torch.tensor(mask,device='cuda:0',dtype=torch.long))

    def generate_batch(self, prompt, length, samples, task, batch_id):
        import torch
        n=len(samples)
        if length not in (256,512) or not 1<=n<=self.settings['batch_size']:
            raise ValueError('Unsupported length or batch size')
        calls=[0]
        def count(*_):
            calls[0]+=1
            if calls[0]>length:raise RuntimeError('Official generation exceeded its fixed-length progress bound')
        handle=self.model.register_forward_pre_hook(count)
        try:
            torch.cuda.synchronize();began=time.perf_counter()
            with torch.no_grad():
                if self.method=='elastic_cache':
                    with elastic_transfers(self.generator) as transfer_counts:
                        canvas, reported_nfe, refresh_fraction = self.generator(self.model,prompt['input_ids'],
                            gen_length=length,window_length=16,mask_id=126336,eos_id=126081,
                            threshold=.90,tokens_per_iter=1,gamma=.90,track_num=1,block_caching=True)
                    output=canvas[:,prompt['input_ids'].shape[1]:]
                else:
                    record=self.generator(self.model,prompt['input_ids'],attention_mask=prompt['attention_mask'],
                        strategy='vanilla',max_new_tokens=length,block_length=32,num_transfer_tokens=1,
                        alg='maskgit_plus',temperature=0.,top_p=None,top_k=None,sigma=0.,threshold=.90,factor=None,
                        stop_until_eos=False,mask_token_id=126336,eos_token_id=126081,pad_token_id=126081,
                        cache_cls=self.cache_cls,ignore_unknown_args='forbid')
            torch.cuda.synchronize();elapsed=time.perf_counter()-began
        finally:handle.remove()
        maximum=None
        if self.method=='d2cache':
            output=record[-1].generated_tokens
            counts=[delta.transfer_index[0].numel() for delta in record.deltas]
            if len(counts)!=calls[0] or sum(counts)!=length:
                raise RuntimeError('Official parallel trajectory accounting mismatch')
            maximum=max(counts)
        else:
            if reported_nfe!=calls[0] or len(transfer_counts)!=calls[0] or not transfer_counts:
                raise RuntimeError('Official Elastic-Cache NFE and transfer accounting mismatch')
            if not 0<sum(transfer_counts)<=length:
                raise RuntimeError('Official Elastic-Cache committed-position count mismatch')
            maximum=max(transfer_counts)
        if tuple(output.shape)!=(n,length):raise RuntimeError('Official batch output shape mismatch')
        result=[]
        for ids,sample in zip(output.tolist(),samples):
            if 126336 in ids:raise RuntimeError('Incomplete official generation')
            text,tokens=postprocess_output(self.tokenizer,ids,sample,task)
            result.append(dict(token_ids=ids,text=text,raw_decoder_text=self.tokenizer.decode(ids,skip_special_tokens=False),
                seconds=elapsed/n,batch_seconds=elapsed,batch_size=n,batch_id=batch_id,
                timing_metric='Batch generation time divided by actual batch size; not single-request latency',
                nfe=calls[0],iterations=calls[0],output_tokens=tokens,
                first_eos=ids.index(126081) if 126081 in ids else None,truncated=126081 not in ids,
                method=self.method,backend=self.settings['backend'],maximum_positions_per_call=maximum,
                reported_layer_refresh_fraction=refresh_fraction if self.method=='elastic_cache' else None))
        verify_upstream(self.source,self.upstream)
        return result


def assert_same_generation(warm, timed):
    for key in ('token_ids', 'text', 'raw_decoder_text', 'nfe', 'iterations', 'first_eos', 'output_tokens'):
        if warm[key] != timed[key]:
            raise RuntimeError('Warm/timed generation differs: ' + key)


def batches(samples,size):
    if size<1:raise ValueError('Positive batch size required')
    return [samples[i:i+size] for i in range(0,len(samples),size)]


def batch_metrics(rows):
    n=len(rows);groups={}
    for row in rows:
        if row['batch_size']<1 or abs(row['seconds']*row['batch_size']-row['batch_seconds'])>1e-9:
            raise RuntimeError('Amortized time does not match actual batch latency')
        value=(row['batch_seconds'],row['batch_size'])
        if row['batch_id'] in groups and groups[row['batch_id']]!=value:
            raise RuntimeError('Inconsistent batch timing metadata')
        groups[row['batch_id']]=value
    seconds=sum(row['seconds'] for row in rows)
    correct=sum(row['correct'] for row in rows)
    return dict(samples=n,correct=correct,accuracy_percent=100*correct/n if n else None,
        mean_seconds=seconds/n if n else None,mean_nfe=sum(row['nfe'] for row in rows)/n if n else None,
        mean_batch_seconds=sum(v[0] for v in groups.values())/len(groups) if groups else None,
        throughput_requests_per_second=n/seconds if seconds else None,batches=len(groups),
        actual_batch_sizes=sorted({v[1] for v in groups.values()}),
        timing_metric='Amortized generation seconds per item; average batch latency reported separately')


def baseline_table(cells):
    headers=['Task','Length','Correct / total','Acc (%)','Time/item (s)','Avg batch (s)','Items/s','NFE/sequence']
    rows=[]
    number=lambda v,d:'N/A' if v is None else f'{v:.{d}f}'
    for key,c in cells.items():
        task,length=key.rsplit('_',1)
        rows.append([task,length,f"{c['correct']}/{c['samples']}",number(c['accuracy_percent'],2),
            number(c['mean_seconds'],3),number(c['mean_batch_seconds'],3),
            number(c['throughput_requests_per_second'],3),number(c['mean_nfe'],2)])
    widths=[max(len(row[i]) for row in [headers,*rows]) for i in range(len(headers))]
    line=lambda row:'| '+' | '.join(value.ljust(widths[i]) for i,value in enumerate(row))+' |'
    return '\n'.join([line(headers),'|'+'|'.join('-'*(w+2) for w in widths)+'|',*(line(row) for row in rows)])
