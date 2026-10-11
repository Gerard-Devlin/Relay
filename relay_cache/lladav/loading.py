"""Fixed visual checkpoint, preprocessing and shared language-weight mapping."""
import copy,hashlib,sys
from pathlib import Path
MODEL_ID="GSAI-ML/LLaDA-V"
VISION_ID="google/siglip2-so400m-patch14-384"
MODEL_REVISION="a10ba1790083e10e07cfcab6e7be93e1823f5792"
VISION_REVISION="e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
MASK_ID=126336

def load_native(cache):
    import torch
    from transformers import AutoTokenizer
    from .model.adapter import LlavaLLaDAConfig,LlavaLLaDAModelLM
    cache=Path(cache);model_path=cache/'models--GSAI-ML--LLaDA-V/snapshots'/MODEL_REVISION
    vision_path=cache/'models--google--siglip2-so400m-patch14-384/snapshots'/VISION_REVISION
    config=LlavaLLaDAConfig.from_pretrained(model_path,local_files_only=True)
    assert config.hidden_size==4096 and config.num_hidden_layers==32 and config.num_attention_heads==32
    assert config.mm_patch_merge_type=='spatial_unpad' and config.mm_resampler_type is None
    config.mm_vision_tower=str(vision_path);config.use_cache=False
    model,info=LlavaLLaDAModelLM.from_pretrained(model_path,config=config,torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,local_files_only=True,attn_implementation='sdpa',output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('LLaDA-V checkpoint did not load exactly: '+repr(info))
    # from_pretrained already assigns BF16 checkpoint parameters. A blanket
    # dtype conversion also rounds non-checkpoint RoPE frequencies, which the
    # official implementation defines and multiplies in FP32.
    model=model.to(device='cuda:0').eval()
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
    for layer in model.model.layers:
        rotary=layer.self_attn.rotary_emb
        rotary.inv_freq=1.0/(rotary.base**(torch.arange(0,rotary.dim,2,dtype=torch.int64,device=model.device).float()/rotary.dim))
        assert rotary.inv_freq.dtype==torch.float32
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=False)
    assert model.model.embed_tokens.weight.shape==(config.vocab_size,config.hidden_size)
    return model,tokenizer,dict(model_revision=MODEL_REVISION,vision_revision=VISION_REVISION,
        loaded_weights='All checkpoint keys matched',precision='BF16 parameters; original FP32 RoPE frequency buffers',vision='official SigLipVisionTower with checkpoint-tuned weights')

def relay_language_from_native(native):
    """Invert the official LLaDA->HF key rename; share all language parameters."""
    import torch
    from relay_cache.llada.loading import load_external
    cls,external=load_external();module=sys.modules[cls.__module__]
    config=cls.config_class.from_pretrained(Path(__file__).parent/'model',local_files_only=True)
    v=native.config
    for name,value in dict(d_model=v.hidden_size,n_layers=v.num_hidden_layers,n_heads=v.num_attention_heads,
        n_kv_heads=v.num_key_value_heads,mlp_hidden_size=v.intermediate_size,vocab_size=v.vocab_size,
        embedding_size=v.vocab_size,rope_theta=v.rope_theta,rms_norm_eps=v.rms_norm_eps,
        max_sequence_length=v.max_position_embeddings,flash_attention=True,use_cache=False).items():setattr(config,name,value)
    mc=module.create_model_config_from_pretrained_config(config);mc.init_device='meta'
    core=module.LLaDAModel(mc,init_params=False);model=cls(config,model=core)
    mapping={'model.embed_tokens.weight':'model.transformer.wte.weight',
        'model.norm.weight':'model.transformer.ln_f.weight','lm_head.weight':'model.transformer.ff_out.weight'}
    suffix={'input_layernorm.weight':'attn_norm.weight','post_attention_layernorm.weight':'ff_norm.weight',
        'self_attn.q_proj.weight':'q_proj.weight','self_attn.k_proj.weight':'k_proj.weight',
        'self_attn.v_proj.weight':'v_proj.weight','self_attn.o_proj.weight':'attn_out.weight',
        'mlp.gate_proj.weight':'ff_proj.weight','mlp.up_proj.weight':'up_proj.weight','mlp.down_proj.weight':'ff_out.weight'}
    for layer in range(32):
        for a,b in suffix.items():mapping[f'model.layers.{layer}.{a}']=f'model.transformer.blocks.{layer}.{b}'
    weights=native.state_dict();assigned={b:weights[a] for a,b in mapping.items()}
    result=model.load_state_dict(assigned,strict=True,assign=True)
    assert not result.missing_keys and not result.unexpected_keys
    target=model.state_dict()
    for a,b in mapping.items():
        assert target[b].data_ptr()==weights[a].data_ptr() and torch.equal(target[b],weights[a]),(a,b)
    assert not any(p.is_meta for p in model.parameters())
    return model.eval(),external,dict(language_parameters_mapped=len(mapping),all_language_weights_byte_identical=True)

def prepare_visual(native,tokenizer,prompt,image):
    import torch
    from .model.processing import process_images,tokenizer_image_token
    from .model.conversation import conv_templates
    from .model.constants import IMAGE_TOKEN_INDEX,IGNORE_INDEX
    if not isinstance(prompt,str) or not prompt:raise ValueError('Explicit prompt text required')
    image=image.convert('RGB');conv=copy.deepcopy(conv_templates['llava_llada']);conv.tokenizer=tokenizer
    conv.append_message(conv.roles[0],'<image>\n'+prompt+' /think');conv.append_message(conv.roles[1],None)
    formatted=conv.get_prompt();ids=tokenizer_image_token(formatted,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(native.device)
    tower=native.get_model().get_vision_tower();pixels=process_images([image],tower.image_processor,native.config)
    pixels=[x.to(device=native.device,dtype=torch.bfloat16) for x in pixels]
    with torch.inference_mode():
        _,_,_,_,embeds,labels=native.prepare_inputs_labels_for_multimodal(ids,None,None,None,ids.clone(),pixels,['image'],image_sizes=[image.size])
    assert labels.shape==embeds.shape[:2] and (labels==IGNORE_INDEX).any()
    virtual=labels.masked_fill(labels==IGNORE_INDEX,0).squeeze(0)
    assert not (virtual==MASK_ID).any() and bool(torch.isfinite(embeds).all())
    return dict(embeds=embeds,virtual_ids=virtual,formatted_prompt=formatted,
        prompt_sha256=hashlib.sha256(formatted.encode()).hexdigest(),visual_positions=int((labels==IGNORE_INDEX).sum()),
        prefix_positions=embeds.shape[1],image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),stop=conv.sep)
