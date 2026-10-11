"""Additional dependencies captured from the historical visual environment."""
import json
from importlib import metadata
from relay_cache.guards import ROOT

def verify_visual_environment():
    expected=json.loads((ROOT/'dllm-eval/configs/lladav_environment.json').read_text())
    actual={name:metadata.version(name) for name in expected['packages']}
    for name,version in expected['packages'].items():
        if actual[name]!=version:raise RuntimeError('Original visual environment version mismatch: '+name)
    return actual
