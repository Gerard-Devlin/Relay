"""Fixed historical visual reproduction settings."""
SETTINGS = {'model': 'GSAI-ML/LLaDA-V', 'revision': 'a10ba1790083e10e07cfcab6e7be93e1823f5792', 'vision_model': 'google/siglip2-so400m-patch14-384', 'vision_revision': 'e8e487298228002f3d8a82e0cd5c8ea9c567f57f', 'precision': 'BF16 parameters / FP32 RoPE frequencies', 'block': 32, 'threshold': 0.9, 'gamma': 0.8, 'track': 4, 'mask': 4, 'history_budget': 128, 'stripe_width': 8, 'seed': 51713, 'lengths': [256, 512], 'verify': False, 'native_visual_priming': True, 'conversation': 'llava_llada', 'prompt_suffix': ' /think', 'scoring': 'deepseek-flash-saved-output-v1'}

def validate_settings(value):
    if value!=SETTINGS:raise ValueError("Frozen LLaDA-V reproduction profile changed")
