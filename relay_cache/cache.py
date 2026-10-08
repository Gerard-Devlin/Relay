"""Relay's immutable-token guards and rotating stripe/boundary cache policy."""
import importlib,torch
from dataclasses import dataclass
from contextlib import contextmanager
from .execution import Frontier,Runtime as NativeRuntime

@dataclass(frozen=True)
class Plan:
    positions:tuple
    optional_rows:tuple
    masked_count:int
    phase:int
    layers:int=32
    width:int=8

    def __post_init__(self):
        assert self.layers==32 and self.width==8 and 0<=self.phase<4
        assert 0<self.masked_count<=len(self.positions)
        assert len(set(self.positions))==len(self.positions)
        assert tuple(sorted(set(self.optional_rows)))==self.optional_rows
        assert all(self.masked_count<=row<len(self.positions) for row in self.optional_rows)

    @property
    def live_rows(self):
        optional=set(self.optional_rows)
        return tuple(i for i in range(len(self.positions)) if i not in optional)

    def compute_rows(self,layer,full=False):
        assert 0<=layer<32
        return tuple(range(len(self.positions))) if full or layer//8==self.phase else self.live_rows

def block_table(masked,total,key_length,block=32):
    assert 0<masked<=total and key_length>0
    def table(start,end):
        return tuple((0,key_length,i,min(i+block,end)) for i in range(start,end,block))
    return table(0,masked),table(masked,total)

def validate_identity(positions,optional_rows,labels,cached_labels,dirty_positions):
    assert len(positions)==len(labels)
    dirty=set(dirty_positions)
    for row in optional_rows:
        pos=positions[row]
        assert pos not in dirty and cached_labels[pos]==labels[row], 'Cannot relay a changed token identity'

def theoretical_row_layers(active,optional):
    assert active>=0 and optional>=0
    return dict(full=(active+optional)*32,relay=active*32+optional*8,
        saved_optional_row_layers=optional*24,
        limitation='Projection/FFN row count, not measured latency or accuracy.')

class RelayFrontier(Frontier):
    def reset(self,*args):
        super().reset(*args);self.optional=None
    def choose(self,decoded,original):
        required=decoded[self.dirty[decoded.long()]]
        optional=original[~self.dirty[original.long()]]
        spare=max(0,128-required.numel())
        self.optional=optional[-spare:] if spare else optional[:0]
        result=torch.cat((required,self.optional))
        self.plans.append(dict(required=required.numel(),optional=self.optional.numel(),allocated=result.numel()))
        return result

class Engine:
    def __init__(self,model,enabled=False,full=False):
        self.model=model;self.enabled=enabled;self.full=full
        self.module=importlib.import_module(type(model.model.transformer.blocks[0]).__module__)
        self.original=self.module.fused_cache_attention
        self.boundaries=None;self.labels=None;self.call=None;self.epoch=0
        self.row_layers=0;self.optional_skipped_row_layers=0;self.normal_calls=0
        self.boundary_bytes=0;self.phase_counts=[0,0,0,0]
    def begin(self,ids,positions,lengths,frontier,masked_count):
        if lengths[-1]:raise ValueError("Unsupported private cache mode")
        if not self.enabled:
            self.call=None;return
        assert len(lengths[5])==1 and self.model.config.n_layers==32
        q=positions[0];assert q.numel()==ids.numel()
        first=self.boundaries is None
        if first:
            n=self.model.model.transformer.blocks[0].k_cache.shape[0]
            self.boundaries=torch.empty((5,n,self.model.config.d_model),dtype=self.model.dtype,device=self.model.device)
            self.labels=torch.full((n,),-1,dtype=torch.long,device=self.model.device)
            self.boundary_bytes=self.boundaries.numel()*self.boundaries.element_size()
        pos=tuple(q.tolist())
        masked_tables=lengths[2].tolist()
        masked_count=sum(end-start for _,_,start,end in masked_tables)
        tracked_optional=getattr(frontier,'optional',None)
        optional=set(tracked_optional.tolist()) if tracked_optional is not None and not first else set()
        rows=tuple(i for i,p in enumerate(pos) if p in optional)
        plan=Plan(pos,rows,int(masked_count),(max(0,self.epoch-1))%4)
        if rows:
            ri=torch.tensor(rows,device=q.device,dtype=torch.long)
            assert torch.equal(self.labels[q[ri].long()],ids[ri]),'Stale token identity at relay input'
            assert not bool(frontier.dirty[q[ri].long()].any()),'Changed token must traverse all32 layers'
        else:ri=torch.empty(0,device=q.device,dtype=torch.long)
        keep=torch.tensor(plan.live_rows,device=q.device,dtype=torch.long)
        short_pos=q.index_select(0,keep)
        masked,tracked=block_table(plan.masked_count,keep.numel(),int(lengths[2][0,1]))
        masked_table=torch.tensor(masked,device=q.device,dtype=torch.int32).reshape(-1,4)
        tracked_table=torch.tensor(tracked,device=q.device,dtype=torch.int32).reshape(-1,4)
        all_table=torch.cat((masked_table,tracked_table),dim=0)
        self.call=dict(plan=plan,first=first,keep=keep,optional=ri,q=q,short_pos=short_pos,
            masked_table=masked_table,tracked_table=tracked_table,all_table=all_table,seen=[])
        self.labels[q.long()]=ids
        self.phase_counts[plan.phase]+=not first;self.normal_calls+=1
    def forward(self,block,x,layer,positions,lengths,softmax_scale=None):
        c=self.call
        if not self.enabled:
            self.row_layers+=x.shape[0]
            return self.original(block,x,layer,positions,lengths,softmax_scale)
        assert c is not None and layer==len(c['seen']);c['seen'].append(layer)
        plan=c['plan'];q=c['q'];full=self.full or c['first'] or layer//8==plan.phase
        if layer%8==0:
            if layer==0:self.boundaries[0].index_copy_(0,q.long(),x)
            if not self.full and not c['first'] and layer//8==plan.phase and c['optional'].numel():
                rows=c['optional'];old=self.boundaries[layer//8].index_select(0,q[rows].long())
                x.index_copy_(0,rows,old)
        if full:
            y=self.original(block,x,layer,positions,lengths,softmax_scale)
            computed=q
        else:
            p=list(positions);p[0]=c['short_pos']
            ls=list(lengths);ls[2]=c['masked_table'];ls[3]=c['tracked_table'];ls[4]=c['all_table']
            small=x.index_select(0,c['keep'])
            updated=self.original(block,small,layer,p,ls,softmax_scale).squeeze(0)
            x.index_copy_(0,c['keep'],updated);y=x.unsqueeze(0)
            computed=c['short_pos'];self.optional_skipped_row_layers+=len(plan.optional_rows)
        self.row_layers+=computed.numel()
        if layer%8==7:
            output=y.squeeze(0) if full else y.squeeze(0).index_select(0,c['keep'])
            self.boundaries[(layer+1)//8].index_copy_(0,computed.long(),output)
        return y
    def end(self):
        if self.call is not None:
            assert self.call['seen']==list(range(32))
            self.epoch+=1;self.call=None
    @contextmanager
    def installed(self):
        assert self.module.fused_cache_attention is self.original
        self.module.fused_cache_attention=self.forward
        try:yield self
        finally:self.module.fused_cache_attention=self.original;self.call=None

class Runtime(NativeRuntime):
    def __init__(self,model,frontier,engine):
        super().__init__(model,frontier);self.engine=engine
    def __call__(self,*args,readout_rows=None,**kwargs):
        self.engine.begin(args[0].squeeze(0),kwargs['positions'],kwargs['lengths'],self.frontier,readout_rows[1])
        try:return super().__call__(*args,readout_rows=readout_rows,**kwargs)
        finally:self.engine.end()
