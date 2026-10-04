"""Consumed-row execution adapter, telemetry and original cache frontier.

Pinned upstream operations stay in model/; local adapters are compiled in memory.
"""
import ast,functools,inspect,io,math,textwrap,torch
import torch.nn.functional as F
from contextlib import contextmanager
from types import SimpleNamespace
def pad_rows(hidden,minimum):
    if minimum<0:
        raise ValueError('Negative head row budget')
    return F.pad(hidden,(0,0,0,max(0,minimum-hidden.shape[1])))

def project(model,hidden,minimum=32):
    count=hidden.shape[1]
    core,tr=model.model,model.model.transformer
    padded=pad_rows(hidden,minimum)
    logits=(F.linear(padded,tr.wte.weight) if core.config.weight_tying else tr.ff_out(padded))[:,:count]
    if core.config.scale_logits:
        logits=logits*(1/math.sqrt(core.config.d_model))
    return logits


@torch.no_grad()
def selected_forward(model,ids,targets,*,minimum=32,**kwargs):
    targets=torch.as_tensor(targets,device=ids.device,dtype=torch.long)
    if targets.ndim!=1 or targets.numel()==0:
        raise ValueError('Nonempty head positions required')
    # Normalize the original full hidden canvas; compact only its head input.
    norm=model.model.transformer.ln_f
    handle=norm.register_forward_hook(lambda _m,_i,value:
        pad_rows(value.index_select(1,targets),minimum))
    try:
        output=model(ids,**kwargs)
        output.logits=output.logits[:,:targets.numel()]
        return output
    finally:
        handle.remove()

def reference(logits, labels=None):
    if logits.ndim!=2 or logits.shape[1]==0:raise ValueError('Nonempty vocabulary required')
    p=logits.double().softmax(-1)
    confidence,top=p.max(-1)
    if labels is not None:
        if labels.shape!=(logits.shape[0],):raise ValueError('One target per row required')
        confidence=p.gather(1,labels[:,None]).flatten()
    return confidence,top

_kernels=None

def kernels():
    # Triton 3.3 resolves JIT symbols through the function module globals.
    # Keep imports lazy for CPU-only tests, but expose modules to its compiler.
    global _kernels,tl,libdevice
    if _kernels is not None:return _kernels
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def partial(Z,M,D,I,V:tl.constexpr,STRIDE:tl.constexpr,CHUNKS:tl.constexpr,BLOCK:tl.constexpr):
        row=tl.program_id(0);chunk=tl.program_id(1)
        column=chunk*BLOCK+tl.arange(0,BLOCK)
        value=tl.load(Z+row*STRIDE+column,column<V,other=-float('inf')).to(tl.float32)
        maximum=tl.max(value,0)
        index=tl.min(tl.where((column<V)&(value==maximum),column,2147483647),0)
        # FP64 CUDA exp and sums, matching the official precision policy.
        denominator=tl.sum(libdevice.exp(value.to(tl.float64)-maximum.to(tl.float64)),0)
        slot=row*CHUNKS+chunk
        tl.store(M+slot,maximum);tl.store(D+slot,denominator);tl.store(I+slot,index)

    @triton.jit
    def finish(Z,L,M,D,I,C,T,V:tl.constexpr,STRIDE:tl.constexpr,CHUNKS:tl.constexpr,
               BLOCK:tl.constexpr,TARGET:tl.constexpr):
        row=tl.program_id(0);offset=tl.arange(0,BLOCK)
        maximum=tl.load(M+row*CHUNKS+offset,offset<CHUNKS,other=-float('inf'))
        totalmax=tl.max(maximum,0)
        sums=tl.load(D+row*CHUNKS+offset,offset<CHUNKS,other=0.)
        denominator=tl.sum(sums*libdevice.exp(maximum.to(tl.float64)-totalmax.to(tl.float64)),0)
        index=tl.load(I+row*CHUNKS+offset,offset<CHUNKS,other=2147483647)
        top=tl.min(tl.where(maximum==totalmax,index,2147483647),0)
        numerator=tl.full((),1.,tl.float64)
        if TARGET:
            label=tl.load(L+row)
            value=tl.load(Z+row*STRIDE+label,(label>=0)&(label<V),other=float('nan')).to(tl.float64)
            numerator=libdevice.exp(value-totalmax.to(tl.float64))
        tl.store(C+row,numerator/denominator);tl.store(T+row,top)

    _kernels=(partial,finish,triton)
    return _kernels

def statistics(logits,labels=None):
    if logits.ndim!=2 or logits.shape[1]==0:raise ValueError('Nonempty vocabulary required')
    if labels is not None and (labels.shape!=(logits.shape[0],) or labels.dtype not in (torch.int32,torch.int64)):
        raise ValueError('One integer target per row required')
    if not logits.is_cuda:return reference(logits,labels)
    if logits.stride(1)!=1:raise ValueError('Contiguous vocabulary axis required')
    if labels is not None and (labels.device!=logits.device or labels.stride(0)!=1):
        raise ValueError('Contiguous targets on logits device required')
    rows,vocabulary=logits.shape
    confidence=torch.empty(rows,device=logits.device,dtype=torch.float64)
    top=torch.empty(rows,device=logits.device,dtype=torch.long)
    if rows==0:return confidence,top
    partial,finish,triton=kernels();block=2048;chunks=triton.cdiv(vocabulary,block)
    maximum=torch.empty((rows,chunks),device=logits.device,dtype=torch.float32)
    denominator=torch.empty((rows,chunks),device=logits.device,dtype=torch.float64)
    indices=torch.empty((rows,chunks),device=logits.device,dtype=torch.int32)
    partial[(rows,chunks)](logits,maximum,denominator,indices,vocabulary,logits.stride(0),chunks,block,num_warps=4)
    finish[(rows,)](logits,labels if labels is not None else top,maximum,denominator,indices,
                   confidence,top,vocabulary,logits.stride(0),chunks,triton.next_power_of_2(chunks),
                   labels is not None,num_warps=4)
    return confidence,top

class Readout:
    """Virtual full-row indexing backed ONLY by the consumed compact rows."""
    def __init__(self, logits, start, count, full_rows):
        if logits.ndim!=3 or logits.shape[0]!=1 or logits.shape[1]<count:
            raise ValueError('Batch-one compact head required')
        if start<0 or count<0 or start+count>full_rows:raise ValueError('Invalid head row range')
        self.logits=logits[0,:count];self.start=start;self.count=count;self.full_rows=full_rows

    def squeeze(self, dim):
        if dim!=0:raise ValueError('Only the pinned generator squeeze(0) is supported')
        return self

    def __getitem__(self, key):
        if not isinstance(key,slice) or key.step not in (None,1):
            raise ValueError('Only consumed contiguous slices are supported')
        start=0 if key.start is None else key.start
        stop=self.full_rows if key.stop is None else key.stop
        if not self.start<=start<=stop<=self.start+self.count:
            raise ValueError('Generator attempted to read an unprojected row')
        return self.logits[start-self.start:stop-self.start]

class ModelReadout:
    def __init__(self, model, *, compact=False, minimum=32, observer=None):
        self.model_ref=model;self.compact=compact;self.minimum=minimum;self.observer=observer
        self.current=None

    def __getattr__(self,name):return getattr(self.model_ref,name)

    def __call__(self,*args,readout_rows=None,**kwargs):
        if readout_rows is None:raise ValueError('Missing pinned consumer row metadata')
        start,count=map(int,readout_rows)
        self.current=dict(start=start,count=count,verify=bool(kwargs['lengths'][-1]))
        if not self.compact:
            value=self.model_ref(*args,**kwargs)
            if self.observer:self.observer(self.current,args,kwargs,value)
            return value
        full=[None]
        def select(_module,_inputs,value):
            full[0]=value.shape[1]
            if not 0<=start<=start+count<=full[0]:raise ValueError('Head consumer outside hidden canvas')
            # Preserve FULL final normalization, then reduce only projection rows.
            # Empty verification has no consumer; Transformer/cache still execute.
            return pad_rows(value[:,start:start+count],self.minimum if count else 0)
        norm=self.model_ref.model.transformer.ln_f
        handle=norm.register_forward_hook(select)
        try:value=self.model_ref(*args,**kwargs)
        finally:handle.remove()
        from types import SimpleNamespace
        return SimpleNamespace(logits=Readout(value.logits,start,count,full[0]))

def generator(function, action_observer=None, *, statistics=None):
    """Fail closed on unknown source; change only consumer metadata/telemetry."""
    original=inspect.unwrap(function)
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1:raise ValueError('Expected one pinned generator')
    functions[0].decorator_list=[]
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model']
    calls.sort(key=lambda n:n.lineno)
    if len(calls)!=2:raise ValueError('Expected exactly normal and verify model calls')
    expressions=('(0, block_m) if verify else (0, query_masked_pos[0].shape[0])',
                 '(seqlen_keep[0] + num_verify, num_verify)')
    for call,expression in zip(calls,expressions):
        if any(k.arg=='readout_rows' for k in call.keywords):raise ValueError('Metadata collision')
        call.keywords.append(ast.keyword(arg='readout_rows',value=ast.parse(expression,mode='eval').body))
    additions=[0]
    reductions=dict(masked=0,verify=0,probability=0)
    class Actions(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            if statistics is not None:
                names=[n.id for n in ast.walk(ast.Tuple(elts=node.targets,ctx=ast.Load()))
                       if isinstance(n,ast.Name)]
                def expected(source):
                    return ast.dump(node,include_attributes=False)==ast.dump(ast.parse(source).body[0],include_attributes=False)
                if names in (['p_masked'],['p_verify']):
                    stem='masked' if names==['p_masked'] else 'verify'
                    if not expected(f'p_{stem} = F.softmax(logits_{stem}_j.to(torch.float64), dim=-1)'):
                        raise ValueError('Unexpected full-vocabulary probability source')
                    reductions['probability']+=1
                    return None
                if names==['x0_p_masked','x0_masked']:
                    if not expected('x0_p_masked, x0_masked = torch.max(p_masked, dim=-1)'):
                        raise ValueError('Unexpected normal confidence source')
                    reductions['masked']+=1
                    node.value=ast.parse('_relay_statistics(logits_masked_j, None)',mode='eval').body
                elif names==['x0_p_verify'] and expected('x0_p_verify = p_verify.gather(1, x_verify_j.unsqueeze(1)).view(-1)'):
                    reductions['verify']+=1
                    node.value=ast.parse('_relay_statistics(logits_verify_j, x_verify_j)[0]',mode='eval').body
            if action_observer is None:return node
            if len(node.targets)!=1:return node
            target=node.targets[0]
            if not (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='x'
                    and isinstance(target.slice,ast.Name) and target.slice.id=='pos_decoded_new_j'):return node
            if not isinstance(node.value,ast.Name) or node.value.id!='x0_decoded_new_j':
                raise ValueError('Unexpected commit source')
            additions[0]+=1
            callback=ast.parse('_relay_action(pos_decoded_new_j, x0_decoded_new_j)').body[0]
            return [node,ast.copy_location(callback,node)]
    tree=Actions().visit(tree)
    if action_observer is not None and additions[0]!=2:raise ValueError('Expected two actual canvas commit sites')
    if statistics is not None:
        if reductions!=dict(masked=2,verify=1,probability=3):
            raise ValueError('Expected exactly three pinned probability consumers')
        if any(isinstance(n,ast.Name) and n.id in ('p_masked','p_verify') for n in ast.walk(tree)):
            raise ValueError('Additional full distribution consumers cannot be bypassed')
    ast.fix_missing_locations(tree)
    scope=dict(original.__globals__)
    if '_relay_action' in scope:raise ValueError('Observer namespace collision')
    scope['_relay_action']=action_observer
    if '_relay_statistics' in scope:raise ValueError('Statistics namespace collision')
    scope['_relay_statistics']=statistics
    exec(compile(tree,original.__code__.co_filename+':consumed_head_rows','exec'),scope)
    return functools.update_wrapper(torch.no_grad()(scope[original.__name__]),original)


@contextmanager
def suppress_official_prints():
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):yield

def adapted(function, action=None, statistics=None, transform=None):
    # Reuse the audited adapter, then modify its executable source AST in memory.
    code = inspect.unwrap(function)
    # generator compiles its changed function under this original filename. It
    # cannot be re-inspected from disk, so build a local source provider for the
    # small boundary fix BEFORE using the audited adapter.
    original_source = textwrap.dedent(inspect.getsource(code))
    tree = ast.parse(original_source)
    fixes = 0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and ast.dump(node,include_attributes=False) == ast.dump(
                ast.parse('logits_masked_j = logits[acc_seqlen_masked : acc_seqlen_masked + block_m]').body[0],
                include_attributes=False):
            node.value.slice.upper = ast.parse('acc_seqlen_masked + min(block_m,query_masked_pos[j].shape[0])',mode='eval').body
            fixes += 1
    assert fixes == 1
    if transform is not None:tree=transform(tree)
    tree.body[0].decorator_list = []
    ast.fix_missing_locations(tree)
    filename = code.__code__.co_filename+':scope_ragged_boundary'
    source = ast.unparse(tree)+'\n'
    import linecache
    linecache.cache[filename] = (len(source),None,source.splitlines(True),filename)
    scope = dict(code.__globals__)
    exec(compile(source,filename,'exec'),scope)
    return generator(torch.no_grad()(scope[code.__name__]),action,statistics=statistics)

def candidate_budget(remaining, ordinary_commits, *, enabled=True):
    if remaining<0 or ordinary_commits<0:raise ValueError('Negative frontier')
    limit=min(24,(64-max(8,ordinary_commits))//2) if enabled else 16
    return min(remaining,max(0,limit))

class Frontier:
    def __init__(self,cache=False,verify=False):
        self.cache=cache;self.verify=verify
    def reset(self,size,device):
        self.dirty=torch.zeros(size,device=device,dtype=torch.bool)
        self.age=0;self.debt=0;self.commits=0;self.plans=[]
    def commit(self,positions,values):
        self.dirty[positions.long()]=True
        self.commits+=positions.numel()
    def refreshed(self,positions):
        self.dirty[positions.long()]=False
    def choose(self,decoded,original):
        if not self.cache:return original
        flag=self.dirty[decoded.long()]
        required=decoded[flag];optional=decoded[~flag]
        self.age+=1;self.debt+=self.commits;self.commits=0
        wide=self.age>=4 or self.debt>=16
        target=128 if wide else (64 if required.numel()>=4 else 32)
        target=max(target,required.numel())
        spare=max(0,target-required.numel())
        result=torch.cat((required,optional[-spare:] if spare else optional[:0]))
        self.plans.append(dict(required=required.numel(),allocated=result.numel(),wide=wide,edit_debt=self.debt))
        if wide:self.age=0;self.debt=0
        return result
    def cap(self,remaining,ordinary):
        return candidate_budget(remaining,ordinary,enabled=self.verify)

def mechanism_generator(function,frontier,statistics):
    """Process-local AST changes at the budget sites; original source untouched."""
    # Add the planner to the original function globals used by the local clone.
    # Restore immediately after compilation; no official disk/source mutation.
    original=__import__('inspect').unwrap(function)
    assert '_scope_frontier' not in original.__globals__
    original.__globals__['_scope_frontier']=frontier
    counts=dict(reset=0,budget=0,cache=0)
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            if len(node.targets)!=1:return node
            target=node.targets[0]
            if isinstance(target,ast.Name) and target.id=='x' and isinstance(node.value,ast.Call):
                if isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='full':
                    counts['reset']+=1
                    reset=ast.parse('_scope_frontier.reset(batch_size*max_length,model.device)').body[0]
                    return [node,ast.copy_location(reset,node)]
            if isinstance(target,ast.Name) and target.id=='num_verify' and ast.dump(node.value,include_attributes=False)==ast.dump(
                    ast.parse('min(num_verify,block_m//2)',mode='eval').body,include_attributes=False):
                counts['budget']+=1
                node.value=ast.parse('_scope_frontier.cap(num_verify,keep_num)',mode='eval').body
            if (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name)
                    and target.value.id=='query_tracked_pos' and isinstance(node.value,ast.Call)
                    and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='cat'):
                # Only the small-budget next-iteration history selection. Full
                # prefill and negative-track full recomputation remain unchanged.
                text=ast.unparse(node.value)
                if 'track_start' in text:
                    counts['cache']+=1
                    node.value.args[0].elts[0]=ast.parse(
                        '_scope_frontier.choose(full_pos[j,:num_decoded[j]],full_pos[j,track_start:num_decoded[j]])',
                        mode='eval').body
            return node
    def transform(tree):
        tree=Rewrite().visit(tree)
        assert counts==dict(reset=1,budget=1,cache=1),counts
        return tree
    try:return adapted(function,frontier.commit,statistics,transform)
    finally:original.__globals__.pop('_scope_frontier')

class Runtime(ModelReadout):
    """Common compact readout and empty-verify bypass for ALL pilot methods."""
    def __init__(self,model,frontier):
        super().__init__(model,compact=True,minimum=32)
        self.frontier=frontier;self.calls=[];self.empty=0
    def __call__(self,*args,readout_rows=None,**kwargs):
        verify=bool(kwargs['lengths'][-1]);start,count=readout_rows
        if verify and count==0:
            self.empty+=1
            # The pinned verification branch is private: no global KV/saliency
            # writes, no logits consumer for S=0. Decoder iteration still runs.
            return SimpleNamespace(logits=Readout(torch.empty((1,0,self.config.vocab_size),
                device=self.device,dtype=self.dtype),start,0,args[0].shape[1]))
        result=super().__call__(*args,readout_rows=readout_rows,**kwargs)
        # ModelReadout has already resolved this metadata to a Python int; do
        # not retain a GPU scalar in the JSON ledger or add a second readback.
        self.calls.append(dict(verify=verify,queries=args[0].shape[1],candidates=result.logits.count if verify else None))
        if not verify:self.frontier.refreshed(kwargs['positions'][0])
        return result

class OutputCapture:
    """Reuse the generator's existing raw decode; never re-tokenize scored text."""
    def __init__(self, tokenizer):self.tokenizer=tokenizer;self.ids=None
    def __getattr__(self,name):return getattr(self.tokenizer,name)
    def __call__(self,*args,**kwargs):return self.tokenizer(*args,**kwargs)
    def decode(self,ids,**kwargs):
        if isinstance(ids,torch.Tensor):ids=ids.tolist()
        if kwargs.get('skip_special_tokens') is False:
            if self.ids is not None:raise ValueError('Unexpected repeated raw readback')
            self.ids=list(ids)
        return self.tokenizer.decode(ids,**kwargs)


@contextmanager
def forbid_sdpa():
    original = torch.nn.functional.scaled_dot_product_attention
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected SDPA call; require pinned fused Flash kernels")
    torch.nn.functional.scaled_dot_product_attention = forbidden
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
