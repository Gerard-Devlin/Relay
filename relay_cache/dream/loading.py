"""Explicit checkpoint download; generation itself stays offline."""
import argparse
import json
from pathlib import Path
MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
REVISION = "05334cb9faaf763692dcf9d8737c642be2b2a6ae"

def validate_checkpoint(path):
    path = Path(path).resolve()
    config = json.loads((path / "config.json").read_text())
    expected = dict(model_type="Dream", num_hidden_layers=28, hidden_size=3584,
                    num_attention_heads=28, num_key_value_heads=4, mask_token_id=151666)
    if any(config.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Unexpected DREAM checkpoint architecture")
    for name in ("configuration_dream.py", "modeling_dream.py", "generation_utils.py",
                 "tokenization_dream.py", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if not (path / name).is_file():
            raise RuntimeError("Incomplete DREAM checkpoint: " + name)
    weights = set(json.loads((path / "model.safetensors.index.json").read_text())["weight_map"].values())
    if not weights:
        raise RuntimeError("Empty DREAM weight index")
    for name in weights:
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise RuntimeError("Invalid DREAM shard name")
        target = (path / name).resolve()
        # HF snapshots normally link to this model's sibling blobs directory.
        cache_blob = (path.parent.name == 'snapshots'
                      and path.parents[1].name == 'models--Dream-org--Dream-v0-Instruct-7B'
                      and target.parent == path.parents[1] / 'blobs')
        if not (target.is_relative_to(path) or cache_blob) or not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("Incomplete or invalid DREAM weight shard")
    return path


def snapshot():
    from huggingface_hub import snapshot_download
    return validate_checkpoint(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))


def load_checkpoint(cls, path, attn_implementation='sdpa'):
    """Preserve official config initialization while obtaining HF loading diagnostics.

    Official DREAM wrappers assume from_pretrained returns only a model; asking
    them for output_loading_info makes that wrapper dereference a tuple.
    """
    import importlib
    import torch
    from transformers import PreTrainedModel
    model, info = PreTrainedModel.from_pretrained.__func__(cls, path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation=attn_implementation, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('DREAM checkpoint loading mismatch: ' + repr(info))
    module = importlib.import_module(cls.__module__.rsplit('.', 1)[0] + '.generation_utils')
    # This is the same assignment made by the upstream from_pretrained wrapper.
    model.generation_config = module.DreamGenerationConfig.from_pretrained(path, local_files_only=True)
    return model


def load_model():
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    path = snapshot()
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    cls = get_class_from_dynamic_module(config.auto_map['AutoModel'], str(path), local_files_only=True)
    model = load_checkpoint(cls, path)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    return model.to("cuda:0").eval(), tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--endpoint', help='Optional Hugging Face mirror, used only for this download')
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(endpoint=args.endpoint).model_info(MODEL_ID, revision=REVISION)
    if info.sha != REVISION:
        raise RuntimeError('Checkpoint revision differs from the pinned official model')
    path = snapshot_download(MODEL_ID, revision=REVISION, cache_dir=args.cache_dir,
        endpoint=args.endpoint, max_workers=2,
        allow_patterns=['*.safetensors', '*.json', '*.py', '*.txt'])
    snapshot = validate_checkpoint(path)
    from safetensors import safe_open
    for name in sorted(set(json.loads((snapshot/'model.safetensors.index.json').read_text())['weight_map'].values())):
        with safe_open(str(snapshot/name), framework='pt', device='cpu') as handle:
            if not list(handle.keys()):
                raise RuntimeError('Empty checkpoint shard')
    print(snapshot)


if __name__ == '__main__':
    main()
