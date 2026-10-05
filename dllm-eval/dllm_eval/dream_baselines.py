"""Thin adapters for pinned, unmodified official DREAM baseline implementations.

Each external method runs in its own process. No sampler/cache source is rewritten.
"""
import importlib
import importlib.util
import os
import sys
import time
import types
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from relay_cache.dream.loading import MODEL_ID, REVISION, snapshot, load_checkpoint
from relay_cache.dream.prompts import prompt_ids
from relay_cache.dream.generate import postprocess_output
from .baseline import official_imports, source_manifest, verify_upstream

METHODS = ('fast_dllm_v1', 'fast_dllm_v1_flash', 'fast_dllm_v1_no_flash', 'd2cache', 'elastic_cache')


def attention_context(method):
    """Force the actual SDPA kernel; unsupported Flash execution raises, never falls back."""
    if method not in ('fast_dllm_v1_flash', 'fast_dllm_v1_no_flash'):
        return nullcontext()
    from torch.nn.attention import sdpa_kernel, SDPBackend
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION if method == 'fast_dllm_v1_flash' else SDPBackend.MATH)


def settings(method):
    if method not in METHODS:
        raise ValueError('Unknown DREAM baseline')
    p = dict(method=method, model=MODEL_ID, revision=REVISION, precision='BF16',
             seed=51713, lengths=[256,512], batch_size=1, threshold=.90,
             prompt_policy='official BOS + prepared paper_prompt; no extra chat template',
             warmup='One excluded full request per prompt before timed replay',
             parallel_decoding=True, temperature=0., top_p=None, top_k=None,
             profile='Pinned official DREAM sampler/cache; common Instruct checkpoint and prepared prompts')
    if method.startswith('fast_dllm_v1'):
        p.update(block_length=32, alg='confidence_threshold', alg_temp=0.,
                 use_cache=True, dual_cache=False, stop_until_eos=False,
                 backend='Official Fast-dLLM DREAM PrefixCache + parallel; SDPA')
        p['attention_policy'] = {'fast_dllm_v1':'auto SDPA',
            'fast_dllm_v1_flash':'SDPA FLASH_ATTENTION only',
            'fast_dllm_v1_no_flash':'SDPA MATH only'}[method]
        p['backend'] += '; ' + p['attention_policy']
    elif method == 'elastic_cache':
        p.update(window_length=32, alg='confidence_threshold', alg_temp=0., gamma=.90,
                 track_num=1, block_caching=True, tokens_per_iter=1, stop_until_eos=True,
                 backend='Official Elastic-Cache DREAM attention tracking')
    else:
        p.update(block_length=32, num_transfer_tokens=1, alg='maskgit_plus',
                 rollout_p=.1, current_k=32, sigma=10., inflate_w=4,
                 generation_sigma=0., stop_until_eos=False,
                 backend='Official d2Cache DREAM eager attention + rollout')
    return p


def official_model(method, source):
    """Import official modules without renaming their files or changing internals."""
    if method == 'd2cache':
        if 'src' in sys.modules:
            raise RuntimeError('Run each official DREAM method in a fresh process')
        deps=source.parent.parent/'deps'
        for name,value in {'MASK_TOKEN_ID':151666,'EOS_TOKEN_ID':151643,'PAD_TOKEN_ID':151643}.items():
            os.environ[name]=str(value)
        with official_imports(source, deps if deps.exists() else None):
            from src.models.dream import DreamModel
            from src.cache import d2Cache
            from src.generation import generate
        return DreamModel, generate, partial(d2Cache, rollout_p=.1,current_k=32,sigma=10.,inflate_w=4)
    is_v1=method.startswith('fast_dllm_v1')
    folder=source/('v1/dream/model' if is_v1 else 'dream/model')
    namespace='_relay_official_'+method+'_dream'
    if namespace in sys.modules:
        raise RuntimeError('Run each official DREAM method in a fresh process')
    spec=importlib.util.spec_from_file_location(namespace,folder/'__init__.py',submodule_search_locations=[str(folder)])
    pkg=importlib.util.module_from_spec(spec);sys.modules[namespace]=pkg
    try:
        spec.loader.exec_module(pkg)
        model=importlib.import_module(namespace+'.modeling_dream').DreamModel
        sampler=importlib.import_module(namespace+('.generation_utils_block' if is_v1 else '.generation_utils_elastic')).DreamGenerationMixin
    except BaseException:
        for key in list(sys.modules):
            if key==namespace or key.startswith(namespace+'.'):sys.modules.pop(key)
        raise
    return model,sampler,None


def generate_official(session, prompt, length):
    """Exactly the official invocation; useful for tiny CPU integration checks too."""
    p=session.settings
    if session.method=='d2cache':
        record=session.generator(session.model,prompt['input_ids'],attention_mask=prompt['attention_mask'],
            strategy='vanilla',max_new_tokens=length,block_length=32,num_transfer_tokens=1,
            alg='maskgit_plus',temperature=0.,top_p=None,top_k=None,sigma=0.,threshold=.90,factor=None,
            stop_until_eos=False,mask_token_id=151666,eos_token_id=151643,pad_token_id=151643,
            cache_cls=session.cache_cls,ignore_unknown_args='forbid')
        return record[-1].generated_tokens
    kwargs=dict(attention_mask=prompt['attention_mask'],max_new_tokens=length,
        output_history=False,return_dict_in_generate=True,steps=length//32,
        temperature=0.,top_p=None,top_k=None,alg='confidence_threshold',alg_temp=0.,threshold=.90)
    if session.method.startswith('fast_dllm_v1'):kwargs.update(block_length=32,dual_cache=False)
    else:kwargs.update(gamma=.90,window_length=32,track_num=1,block_caching=True,tokens_per_iter=1,
                        eos_id=session.tokenizer.eos_token_id,bos_id=session.tokenizer.bos_token_id)
    result=session.model.diffusion_generate(prompt['input_ids'],**kwargs)
    return result.sequences[:,prompt['input_ids'].shape[1]:]


class Session:
    def __init__(self,method,source):
        import torch
        from transformers import AutoTokenizer
        self.method,self.source=method,Path(source).resolve()
        self.settings=settings(method);self.upstream=source_manifest(method,self.source)
        torch.set_num_threads(1);torch.manual_seed(51713)
        cls,sampler,self.cache_cls=official_model(method,self.source)
        path=snapshot()
        self.model=load_checkpoint(cls,path,'eager' if method=='d2cache' else 'sdpa')
        self.model=self.model.to('cuda:0').eval()
        if method.startswith('fast_dllm_v1') and any(type(layer.self_attn).__name__ != 'DreamSdpaAttention'
                                                   for layer in self.model.model.layers):
            raise RuntimeError('Official DREAM v1 attention path changed; backend labels cannot be trusted')
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=True)
        if method!='d2cache':
            # Same two method bindings as each upstream eval.py.
            self.model.diffusion_generate=types.MethodType(sampler.diffusion_generate,self.model)
            self.model._sample=types.MethodType(sampler._sample,self.model)
        self.generator=sampler

    def prepare_batch(self,texts,task):
        import torch
        if len(texts)!=1:raise ValueError('DREAM official comparison is batch one')
        ids=torch.tensor([prompt_ids(self.tokenizer,texts[0])],device=self.model.device,dtype=torch.long)
        return dict(input_ids=ids,attention_mask=ids.ne(self.tokenizer.pad_token_id))

    def generate_batch(self,prompt,length,samples,task,batch_id):
        import torch
        if length not in (256,512) or len(samples)!=1:raise ValueError('Fixed DREAM profile requires batch one and length 256/512')
        calls=[0]
        def count(*_):
            calls[0]+=1
            if calls[0]>length+length//32:raise RuntimeError('DREAM generation exceeded progress bound')
        handle=self.model.register_forward_pre_hook(count)
        try:
            with torch.no_grad(),torch.random.fork_rng(devices=[self.model.device.index or 0]),attention_context(self.method):
                torch.manual_seed(51713);torch.cuda.synchronize();start=time.perf_counter()
                output=generate_official(self,prompt,length)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
        finally:handle.remove()
        if tuple(output.shape)!=(1,length) or not calls[0]:raise RuntimeError('Official DREAM output shape or NFE mismatch')
        ids=output[0].tolist();eos=self.tokenizer.eos_token_id
        first=ids.index(eos) if eos in ids else None
        # Elastic legitimately leaves MASKs after its first EOS. Never fill them artificially.
        valid=ids[:first] if first is not None and self.settings['stop_until_eos'] else ids
        if 151666 in valid:raise RuntimeError('Unfinished DREAM output before stopping boundary')
        raw=self.tokenizer.decode(ids,skip_special_tokens=False)
        text,processed=postprocess_output(self.tokenizer,raw,samples[0])
        verify_upstream(self.source,self.upstream)
        return [dict(token_ids=ids,text=text,raw_decoder_text=raw,seconds=elapsed,batch_seconds=elapsed,
            batch_size=1,batch_id=batch_id,nfe=calls[0],iterations=calls[0],output_tokens=len(processed),
            first_eos=first,truncated=first is None,method=self.method,model=MODEL_ID,revision=REVISION,
            backend=self.settings['backend'],timing_metric='Warmed prepared-prompt batch-one generation latency')]
