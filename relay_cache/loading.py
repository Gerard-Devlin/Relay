"""Load only the pinned Flash execution class and original offline checkpoint."""
import importlib,os,sys
from pathlib import Path
MODEL_ID="GSAI-ML/LLaDA-8B-Instruct"
REVISION="08b83a6feb34df1a6011b80c3c00c7563e963b07"
MASK_ID=126336
ROOT=Path(__file__).resolve().parent.parent

def snapshot():
    from huggingface_hub import snapshot_download
    try:
        path=Path(snapshot_download(MODEL_ID,revision=REVISION,local_files_only=True))
    except Exception:
        cache=Path(os.environ.get("HF_HUB_CACHE",Path(os.environ.get("HF_HOME","~/.cache/huggingface"))/"hub"))
        path=cache.expanduser()/"models--GSAI-ML--LLaDA-8B-Instruct"/"snapshots"/REVISION
    if not (path/"config.json").is_file():raise RuntimeError("Pinned LLaDA checkpoint is incomplete")
    return path

def load_external():
    # Preserve original module ABI while keeping descriptive public filenames.
    source=ROOT/"relay_cache/model"
    parent=source.parent
    for name in ("model","flash_cache_triton","_relay_decoder"):
        module=sys.modules.get(name)
        if module is not None:
            locations=[getattr(module,"__file__",None),*getattr(module,"__path__",[])]
            if not any(p and Path(p).resolve().is_relative_to(source) for p in locations):
                raise RuntimeError("Conflicting external module: "+name)
    def pinned_module(name,path):
        if name not in sys.modules:
            spec=importlib.util.spec_from_file_location(name,path)
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module
            try:spec.loader.exec_module(module)
            except BaseException:sys.modules.pop(name,None);raise
        return sys.modules[name]
    pinned_module('flash_cache_triton',source/'kernels.py')
    if str(parent) not in sys.path:sys.path.insert(0,str(parent))
    model=importlib.import_module('model.modeling_llada')
    decoder=pinned_module('_relay_decoder',source/'decoder.py')
    return model.LLaDAModelLM,decoder.generate_with_Flash_dLLM


def load_model():
    import torch
    from transformers import AutoTokenizer
    cls,external=load_external();checkpoint=snapshot()
    config=cls.config_class.from_pretrained(checkpoint,local_files_only=True)
    config.flash_attention=True
    model,info=cls.from_pretrained(checkpoint,config=config,local_files_only=True,
        torch_dtype=torch.bfloat16,output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"],info
    return model.to("cuda:0").eval(),AutoTokenizer.from_pretrained(checkpoint,local_files_only=True),external
