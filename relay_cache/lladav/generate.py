"""The historical full-evaluated visual Relay decoder; no research switches."""
import contextlib,time
from .loading import load_native,relay_language_from_native,prepare_visual
FORBIDDEN=(126081,126080,126346,126347)

class Session:
    def __init__(self,cache):
        import torch
        torch.set_num_threads(1);torch.manual_seed(51713)
        self.native,self.tokenizer,self.loading=load_native(cache)
        self.model,self.external,self.mapping=relay_language_from_native(self.native)
    def prepare(self,prompt,image):
        return prepare_visual(self.native,self.tokenizer,prompt,image)
    def generate(self,value,length,audit=False):
        if length not in (256,512):raise ValueError('Frozen profile supports 256/512')
        return self.relay(value,length,audit)
    def relay(self,p,length,audit):
        import torch
        from relay_cache.cache import Engine,Runtime
        from .priming import native_visual_priming
        from relay_cache.llada.generate import RelayCounter
        from relay_cache.execution import mechanism_generator,statistics,OutputCapture,forbid_sdpa,suppress_official_prints
        frontier=RelayCounter(cache=True);engine=Engine(self.model,True,False);runtime=Runtime(self.model,frontier,engine)
        function=mechanism_generator(self.external,frontier,statistics)
        if audit:
            old_reset=frontier.reset
            def reset(*args):old_reset(*args);frontier.actions=[]
            frontier.reset=reset
        def inject(_model,args,kwargs):
            ids=args[0];pos=kwargs['positions'][0].long();embedded=self.model.model.transformer.wte(ids)
            rows=(pos<p['prefix_positions']).nonzero().flatten()
            if rows.numel():embedded[:,rows]=p['embeds'][:,pos[rows]]
            kwargs['inputs_embeds']=embedded;return args,kwargs
        def restrict(_model,args,output):output.logits[...,list(FORBIDDEN)]=-float('inf');return output
        a=self.model.register_forward_pre_hook(inject,with_kwargs=True);b=self.model.register_forward_hook(restrict)
        capture=OutputCapture(self.tokenizer);texts=[None];steps=[0]
        try:
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode(),engine.installed(),native_visual_priming(engine,self.native),forbid_sdpa(),suppress_official_prints():
                function(runtime,[p['virtual_ids']],[p['prefix_positions']],1,texts,steps,gen_length=length,block_length=32,
                    threshold=.9,track_num=4,mask_num=4,tokenizer=capture,stop_tokens=[p['stop']])
            torch.cuda.synchronize();seconds=time.perf_counter()-start
        finally:a.remove();b.remove()
        assert capture.ids is not None and len(runtime.calls)>0
        assert engine.native_priming_layers==32
        row=dict(token_ids=capture.ids,text=self.tokenizer.decode(capture.ids,skip_special_tokens=True),seconds=seconds,
            nfe=len(runtime.calls),iterations=steps[0],actual_row_layers=engine.row_layers,boundary_bytes=engine.boundary_bytes,
            native_priming_layers=engine.native_priming_layers,
            sampling=dict(block=32,threshold=.9,threshold_meaning='Relay confidence threshold',history=128,stripe=8,verify=False),
            backend='Official native visual cache priming, then existing Relay fused Triton execution',truncated=126081 not in capture.ids)
        if audit:row['actions']=frontier.actions
        return row
