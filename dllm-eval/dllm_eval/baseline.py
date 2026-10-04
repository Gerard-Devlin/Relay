"""Thin adapters for unmodified, pinned ES-dLLM and SparseD implementations."""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from relay_cache.utils import sha256, prompt_ids
from relay_cache.llada.loading import MODEL_ID, REVISION, snapshot
from relay_cache.llada.generate import postprocess_output

SOURCES = {
    'es_dllm': ('https://github.com/zhuzj19/ES-dLLM.git', '3c5151d44d88354925a2f1b4a75745aee3444f36'),
    'sparsed': ('https://github.com/INV-WZQ/SparseD.git', '52155b4aa78368695f96744cade06d2e0865d085'),
}
SPARSE_KERNEL_OPTIONS = dict(BLOCK_M=64, BLOCK_N=64, num_stages=2)


def configure_sparse_kernel(model_module):
    """Keep upstream mask/math; cap launch tiles for consumer-GPU shared memory."""
    original = model_module.flex_attn
    def compatible(*args, **kwargs):
        if 'kernel_options' in kwargs:
            raise RuntimeError('Unexpected upstream kernel-options override')
        return original(*args, **kwargs, kernel_options=dict(SPARSE_KERNEL_OPTIONS))
    model_module.flex_attn = compatible


def settings(method):
    result = dict(method=method, model=MODEL_ID, revision=REVISION, precision='BF16',
                  seed=51713, lengths=[256, 512], sampling='official deterministic low_confidence',
                  warmup='One excluded full request per prompt before timed replay', batch_size=1)
    if method == 'es_dllm':
        result.update(profile='official eval_instruct_singlegpu HiddenState',
                      ESdLLM_mode='HiddenState', importance_score_alpha=.5,
                      proportion_steps=[[1, 0], [.5, .125], [.25, .25]],
                      use_kvcache=True, parallel_mode=False, delay_eos_generation=True,
                      per_task={'gsm8k': [64, 64, 16], 'humaneval': [64, 64, 4],
                                'mbpp': [64, 64, 4], 'math': [256, 256, 8]})
    elif method == 'sparsed':
        result.update(profile='official README short_context', block=32, skip=.2, select=.5,
                      attention_block_size=32, steps='gen_length', cache=False,
                      flex_kernel_options=dict(SPARSE_KERNEL_OPTIONS))
    else:
        raise ValueError('Unknown external method')
    return result


def source_manifest(method, source):
    source = Path(source).resolve()
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    url, revision = SOURCES[method]
    if actual != revision:
        raise RuntimeError('Upstream revision mismatch')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Upstream tracked files modified')
    files = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*.py')
             if '.git' not in p.parts and '__pycache__' not in p.parts}
    return dict(url=url, revision=revision, files=files)


def verify_upstream(source, manifest):
    source = Path(source)
    current = {p.relative_to(source).as_posix() for p in source.rglob('*.py')
               if '.git' not in p.parts and '__pycache__' not in p.parts}
    if current != set(manifest['files']):
        raise RuntimeError('Upstream source file set changed')
    for name, digest in manifest['files'].items():
        if sha256(source / name) != digest:
            raise RuntimeError('Upstream source changed: ' + name)


def module(name, path, package=False):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path,
            submodule_search_locations=[str(path.parent)] if package else None)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    try:
        spec.loader.exec_module(result)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return result


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
        if method == 'es_dllm':
            # Official absolute imports are isolated by one method per worker process.
            if 'models' in sys.modules or 'early_skipping' in sys.modules:
                raise RuntimeError('Conflicting upstream module namespace')
            sys.path.insert(0, str(self.source))
            import models.hook_model as hooks
            self.generator = module('_es_dllm_generator', self.source / 'generate.py').batch_generate
            self.model, info = AutoModel.from_pretrained(checkpoint, trust_remote_code=True,
                    local_files_only=True, torch_dtype=torch.bfloat16, output_loading_info=True)
            hooks.transform_llada_model(self.model)
        else:
            package = module('_sparsed_llada', self.source / 'models/LLaDA/__init__.py', package=True)
            cls = package.LLaDAModelLM
            configure_sparse_kernel(sys.modules[cls.__module__])
            config = cls.config_class.from_pretrained(checkpoint, local_files_only=True)
            # Preserve the official dense dispatcher (FlashAttention when available).
            config.flash_attention = True
            self.model, info = cls.from_pretrained(checkpoint, config=config, local_files_only=True,
                    torch_dtype=torch.bfloat16, output_loading_info=True)
            self.generator = package.generate
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError('Original checkpoint does not match official model class: ' + repr(info))
        self.model = self.model.to('cuda:0').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

    def prepare(self, text, task):
        import torch
        return torch.tensor(prompt_ids(self.tokenizer, text, task, preformatted=True),
                            device='cuda:0', dtype=torch.long).unsqueeze(0)

    def generate(self, prompt, length, sample, task):
        import torch
        if length not in (256, 512):
            raise ValueError('Unsupported evaluation length')
        calls = [0]
        handle = self.model.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0]+1))
        try:
            torch.cuda.synchronize()
            began = time.perf_counter()
            with torch.no_grad():
                if self.method == 'es_dllm':
                    block, prompt_freq, block_freq = self.settings['per_task'][task]
                    kwargs = dict(gen_length=length, block_length=block, temperature=0., cfg_scale=0.,
                        use_kvcache=True, parallel_mode=False, token_per_step=1, threshold=None,
                        print_log=False, record_time=False, statistics=False, delay_eos_generation=True,
                        sparse_kv=1., delay_step=-1, ESdLLM_mode='HiddenState', importance_score_alpha=.5,
                        prompt_update_freq=prompt_freq, block_update_freq=block_freq,
                        proportion_steps=self.settings['proportion_steps'])
                    output, _ = self.generator(self.model, prompt, None, kwargs)
                else:
                    params = dict(skip=.2, select=.5, block_size=32, new_generation=length, whole_steps=length)
                    output = self.generator(self.model, prompt, steps=length, gen_length=length,
                        block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence', SparseD_param=params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
        finally:
            handle.remove()
        ids = output[0, prompt.shape[1]:].tolist()
        if len(ids) != length or 126336 in ids or calls[0] != length:
            raise RuntimeError('Incomplete generation or unexpected official NFE')
        text, output_tokens = postprocess_output(self.tokenizer, ids, sample, task)
        verify_upstream(self.source, self.upstream)
        return dict(token_ids=ids, text=text, raw_decoder_text=self.tokenizer.decode(ids, skip_special_tokens=False),
                    seconds=elapsed, nfe=calls[0], iterations=calls[0], output_tokens=output_tokens,
                    first_eos=ids.index(126081) if 126081 in ids else None, truncated=126081 not in ids,
                    method=self.method, backend='Official SDPA + cache hooks' if self.method=='es_dllm' else 'Official FlexAttention + official dense attention dispatch')


def assert_same_generation(warm, timed):
    for key in ('token_ids', 'text', 'raw_decoder_text', 'nfe', 'iterations', 'first_eos', 'output_tokens'):
        if warm[key] != timed[key]:
            raise RuntimeError('Warm/timed generation differs: ' + key)
