"""Official visual-model cache priming; subsequent Relay execution is unchanged.

The first Relay call already computes every canvas row through all layers.
Execute that work with the checkpoint's own decoder layers and retain the Q/K/V
they actually computed. This avoids changing the native visual model's BF16
arithmetic during priming. No teacher forward or future output is available.
"""
import contextlib

@contextlib.contextmanager
def native_visual_priming(engine,native):
    import torch
    import importlib
    module=importlib.import_module(type(native.model.layers[0].self_attn).__module__)
    original=engine.original
    native_sdpa=torch.nn.functional.scaled_dot_product_attention
    fixed={}
    def prime(block,x,layer,positions,lengths,softmax_scale=None):
        if not engine.call['first']:
            return original(block,x,layer,positions,lengths,softmax_scale)
        q=positions[0].long();n=q.numel()
        if layer==0:
            if not torch.equal(q.sort().values,torch.arange(n,device=q.device)):
                raise RuntimeError('Native visual priming requires the full current canvas')
            fixed['positions']=torch.arange(n,device=q.device).unsqueeze(0)
            fixed['mask']=native.model._update_causal_mask(None,x.unsqueeze(0),fixed['positions'][0],is_causal=False)
        # Use native position order for all BF16 GEMMs, including the MLP.
        natural=torch.empty_like(x);natural.index_copy_(0,q,x)
        attention=native.model.layers[layer].self_attn;captured={};handles=[]
        def keep(name):
            def hook(_m,args,value):captured[name]=value
            return hook
        for name in ('q','k','v'):
            handles.append(getattr(attention,name+'_proj').register_forward_hook(keep(name)))
        blocked_sdpa=torch.nn.functional.scaled_dot_product_attention
        try:
            torch.nn.functional.scaled_dot_product_attention=native_sdpa
            output=native.model.layers[layer](natural.unsqueeze(0),attention_mask=fixed['mask'],
                position_ids=fixed['positions'],use_cache=False,output_attentions=False,
                cache_position=fixed['positions'][0])[0]
        finally:
            torch.nn.functional.scaled_dot_product_attention=blocked_sdpa
            for handle in handles:handle.remove()
        heads=attention.num_heads;dim=attention.head_dim
        query=captured['q'].view(1,n,heads,dim).transpose(1,2)
        key=captured['k'].view(1,n,attention.num_key_value_heads,dim).transpose(1,2)
        value=captured['v'].view(1,n,attention.num_key_value_heads,dim).transpose(1,2)
        cos,sin=attention.rotary_emb(value,fixed['positions'])
        query,key=module.apply_rotary_pos_emb(query,key,cos,sin)
        block.k_cache[:n].copy_(key.transpose(1,2).reshape(n,-1))
        block.v_cache[:n].copy_(value.transpose(1,2).reshape(n,-1))
        engine.native_priming_layers=getattr(engine,'native_priming_layers',0)+1
        # Retain the same unnormalized QK saliency definition. Only MASK
        # queries contribute, with the original block/head normalization.
        if layer>1:
            importance=positions[4]
            denominator=lengths[8]*heads
            for _,end,start,stop in lengths[2].tolist():
                selected=q[start:stop]
                scores=torch.matmul(query[0].index_select(1,selected).float(),key[0,:,:end].float().transpose(-1,-2))
                importance[:end].add_(scores.sum((0,1))/denominator)
        return output.squeeze(0).index_select(0,q).unsqueeze(0)
    engine.original=prime
    try:yield
    finally:engine.original=original
