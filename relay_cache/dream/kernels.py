"""DREAM RoPE and KV updates with persistent grouped-head mirrors.

Projection GEMMs and attention are unchanged. The fused update preserves
the separate BF16 rounding of each PyTorch RoPE multiplication and addition.
"""
_kernel = None


def kernel():
    global _kernel,tl
    if _kernel is not None:return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def update(Q,K,V,P,C,S,QO,KC,VC,KE,VE,
               QS_H:tl.constexpr,QS_R:tl.constexpr,KS_H:tl.constexpr,KS_R:tl.constexpr,
               VS_H:tl.constexpr,VS_R:tl.constexpr,CS_R:tl.constexpr,SS_R:tl.constexpr,
               R,N,D:tl.constexpr,G:tl.constexpr,BM:tl.constexpr):
        row=tl.program_id(0)*BM+tl.arange(0,BM)
        head=tl.program_id(1);kvhead=head//G
        dim=tl.arange(0,D)
        other=tl.where(dim<D//2,dim+D//2,dim-D//2)
        sign=tl.where(dim<D//2,-1.,1.)
        pos=tl.load(P+row,row<R,other=0)
        cosine=tl.load(C+pos[:,None]*CS_R+dim[None,:],row[:,None]<R,other=0).to(tl.float32)
        sine=tl.load(S+pos[:,None]*SS_R+dim[None,:],row[:,None]<R,other=0).to(tl.float32)
        q=tl.load(Q+head*QS_H+row[:,None]*QS_R+dim[None,:],row[:,None]<R,other=0).to(tl.float32)
        qr=tl.load(Q+head*QS_H+row[:,None]*QS_R+other[None,:],row[:,None]<R,other=0).to(tl.float32)*sign[None,:]
        k=tl.load(K+kvhead*KS_H+row[:,None]*KS_R+dim[None,:],row[:,None]<R,other=0).to(tl.float32)
        kr=tl.load(K+kvhead*KS_H+row[:,None]*KS_R+other[None,:],row[:,None]<R,other=0).to(tl.float32)*sign[None,:]
        dtype=Q.dtype.element_ty
        # Match q*cos + rotate_half(q)*sin: round each product first.
        qo=(q*cosine).to(dtype).to(tl.float32)+(qr*sine).to(dtype).to(tl.float32)
        ko=(k*cosine).to(dtype).to(tl.float32)+(kr*sine).to(dtype).to(tl.float32)
        value=tl.load(V+kvhead*VS_H+row[:,None]*VS_R+dim[None,:],row[:,None]<R,other=0)
        tl.store(QO+head*R*D+row[:,None]*D+dim[None,:],qo,row[:,None]<R)
        tl.store(KE+head*N*D+pos[:,None]*D+dim[None,:],ko,row[:,None]<R)
        tl.store(VE+head*N*D+pos[:,None]*D+dim[None,:],value,row[:,None]<R)
        if head%G==0:
            tl.store(KC+kvhead*N*D+pos[:,None]*D+dim[None,:],ko,row[:,None]<R)
            tl.store(VC+kvhead*N*D+pos[:,None]*D+dim[None,:],value,row[:,None]<R)
    _kernel=(update,triton)
    return _kernel


def prepare(q,k,v,rows,cos,sin,caches,expanded,sequence_length,rotate,repeat):
    import torch
    batch,heads,count,dim=q.shape
    kvheads=k.shape[1];groups=heads//kvheads
    if batch!=1 or heads%kvheads or k.shape!=v.shape or k.shape[2:]!=(count,dim):
        raise ValueError('DREAM batch-one grouped KV shapes required')
    if caches is None:
        if count!=sequence_length:raise RuntimeError('Initialize every KV position')
        caches=tuple(torch.empty((1,kvheads,sequence_length,dim),device=q.device,dtype=q.dtype) for _ in range(2))
        expanded=tuple(torch.empty((1,heads,sequence_length,dim),device=q.device,dtype=q.dtype) for _ in range(2))
    assert expanded is not None
    fused=q.is_cuda and q.dtype in (torch.bfloat16,torch.float16) and all(
        t.dtype==q.dtype for t in (k,v,cos,sin)) and dim in (64,128,256)
    if fused:
        out=torch.empty((1,heads,count,dim),device=q.device,dtype=q.dtype)
        update,triton=kernel()
        update[(triton.cdiv(count,4),heads)](q,k,v,rows,cos,sin,out,*caches,*expanded,
            q.stride(1),q.stride(2),k.stride(1),k.stride(2),v.stride(1),v.stride(2),
            cos.stride(1),sin.stride(1),count,sequence_length,dim,groups,4,
            num_warps=4,enable_fp_fusion=False)
    else:
        out,k=rotate(q,k,cos.index_select(1,rows),sin.index_select(1,rows))
        caches[0].index_copy_(2,rows,k);caches[1].index_copy_(2,rows,v)
        expanded[0].index_copy_(2,rows,repeat(k,groups))
        expanded[1].index_copy_(2,rows,repeat(v,groups))
        out=out.contiguous()
    return out,caches,expanded
