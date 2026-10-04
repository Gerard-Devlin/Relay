"""Evaluation adapters for unmodified, pinned dLLM-cache and d2Cache sources."""
import importlib.util
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
    'dllm_cache': ('https://github.com/maomaocun/dLLM-cache.git', '17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a'),
    'd2cache': ('https://github.com/Kamichanw/d2Cache.git', '216b4557f4baf318a246763af805f414cbfb21a4'),
}
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
    if method == 'dllm_cache':
        result.update(profile='Official scripts: first cached LLaDA-Instruct command for each task',
                      transfer_ratio=.25, cfg_interval_steps=1, cfg_scale=0., steps='gen_length',
                      per_task={'gsm8k': [8, 50, 7], 'humaneval': [32, 50, 8],
                                'mbpp': [32, 100, 5], 'math': [256, 50, 1]},
                      backend='Official original-checkpoint attention dispatcher + cache hooks')
    elif method == 'd2cache':
        result.update(profile='Official configs/gen_args.py d2cache LLaDA-Instruct defaults',
                      rollout_p=.1, current_k=32, sigma=10., inflate_w=0,
                      block='gen_length', num_transfer_tokens=1, alg='maskgit_plus',
                      backend='Official eager attention with attention rollout',
                      cli_dependencies=dict(omegaconf='2.3.0', hydra_core='1.3.2', antlr='4.9.3'))
    else:
        raise ValueError('Unknown external method')
    return result


def files(source):
    return {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*')
            if p.is_file() and p.suffix in SOURCE_SUFFIXES
            and '.git' not in p.parts and '__pycache__' not in p.parts}


def source_manifest(method, source):
    source = Path(source).resolve()
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    url, revision = SOURCES[method]
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
        if method == 'dllm_cache':
            if 'dllm_cache' in sys.modules:
                raise RuntimeError('Conflicting upstream module namespace')
            with official_imports(self.source):
                from dllm_cache.cache import dLLMCache
                from dllm_cache.hooks import register_cache_LLaDA
                self.cache_cls = dLLMCache
                self.register_cache = register_cache_LLaDA
                self.generator = module('_official_dllm_generate', self.source / 'utils/generate_function.py').generate
            self.model, info = AutoModel.from_pretrained(checkpoint, trust_remote_code=True,
                    local_files_only=True, torch_dtype=torch.bfloat16, output_loading_info=True)
            self.register_cache(self.model, 'model.transformer.blocks')
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
                self.cache_cls = partial(d2Cache, rollout_p=.1, current_k=32, sigma=10., inflate_w=0)
            self.model, info = LLaDAModelLM.from_pretrained(checkpoint, local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation='eager', output_loading_info=True)
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError('Original checkpoint does not match official model class: ' + repr(info))
        self.model = self.model.to('cuda:0').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.current_task = None

    def prepare(self, text, task):
        import torch
        return torch.tensor(prompt_ids(self.tokenizer, text, task, preformatted=True),
                            device='cuda:0', dtype=torch.long).unsqueeze(0)

    def generate(self, prompt, length, sample, task):
        import torch
        if length not in (256, 512):
            raise ValueError('Unsupported evaluation length')
        if self.method == 'dllm_cache':
            block, prompt_interval, gen_interval = self.settings['per_task'][task]
            if self.current_task != task:
                self.cache_cls.new_instance(prompt_interval_steps=prompt_interval,
                    gen_interval_steps=gen_interval, cfg_interval_steps=1, transfer_ratio=.25)
                self.current_task = task
        calls = [0]
        handle = self.model.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0]+1))
        try:
            torch.cuda.synchronize()
            began = time.perf_counter()
            with torch.no_grad():
                if self.method == 'dllm_cache':
                    output = self.generator(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                        model=self.model, steps=length, gen_length=length, block_length=block,
                        temperature=0., cfg_scale=0., remasking='low_confidence')
                else:
                    record = self.generator(self.model, prompt, strategy='vanilla',
                        max_new_tokens=length, block_length=length, num_transfer_tokens=1,
                        alg='maskgit_plus', temperature=0., top_p=None, top_k=None, sigma=10.,
                        threshold=None, factor=None, stop_until_eos=False,
                        mask_token_id=126336, eos_token_id=126081, pad_token_id=126081,
                        cache_cls=self.cache_cls, ignore_unknown_args='forbid')
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
        finally:
            handle.remove()
        if self.method == 'd2cache':
            # Official DecodeRecord reconstruction is output processing, outside timing.
            output = record[-1].generated_tokens
        ids = output[0].tolist()
        if len(ids) != length or 126336 in ids or calls[0] != length:
            raise RuntimeError('Incomplete generation or unexpected official NFE: ' + repr((len(ids), calls[0])))
        text, output_tokens = postprocess_output(self.tokenizer, ids, sample, task)
        verify_upstream(self.source, self.upstream)
        return dict(token_ids=ids, text=text, raw_decoder_text=self.tokenizer.decode(ids, skip_special_tokens=False),
                    seconds=elapsed, nfe=calls[0], iterations=calls[0], output_tokens=output_tokens,
                    first_eos=ids.index(126081) if 126081 in ids else None, truncated=126081 not in ids,
                    method=self.method, backend=self.settings['backend'])


def assert_same_generation(warm, timed):
    for key in ('token_ids', 'text', 'raw_decoder_text', 'nfe', 'iterations', 'first_eos', 'output_tokens'):
        if warm[key] != timed[key]:
            raise RuntimeError('Warm/timed generation differs: ' + key)
