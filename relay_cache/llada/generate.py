"""The unchanged historical warmed prepared-prompt request path."""
import hashlib,time
from ..cache import Engine,Runtime,RelayFrontier
from ..execution import Frontier,mechanism_generator
from ..execution import statistics
from ..execution import suppress_official_prints
from ..execution import OutputCapture,forbid_sdpa
from ..utils import prompt_ids
EOS_ID=126081

def postprocess_output(tokenizer, token_ids, sample, task):
    """Mirror Fast-dLLM v1's lm-eval text truncation and speed token count."""
    if task == "humaneval":
        output_tokens = sum(token != EOS_ID for token in token_ids)
        return tokenizer.decode(token_ids, skip_special_tokens=True), output_tokens
    generation_kwargs = sample.get("generation_kwargs")
    if not generation_kwargs:
        return tokenizer.decode(token_ids, skip_special_tokens=True), len(token_ids)
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    for stop in generation_kwargs.get("until", []):
        if stop in text:
            text = text.split(stop, 1)[0]
    processed_ids = tokenizer(text)["input_ids"]
    output_tokens = sum(token != EOS_ID for token in processed_ids)
    return tokenizer.decode(processed_ids, skip_special_tokens=True), output_tokens


class Counting:
    def reset(self,*args):
        super().reset(*args);self.stage=False;self.ordinary_accepted=0;self.private_accepted=0;self.actions=None
    def commit(self,positions,values):
        super().commit(positions,values)
        if self.stage:self.private_accepted+=positions.numel()
        else:self.ordinary_accepted+=positions.numel()
        if self.actions is not None:self.actions.append((tuple(positions.tolist()),tuple(values.tolist())))

class Counter(Counting,Frontier):pass

class RelayCounter(Counting,RelayFrontier):pass

class Session:
    def __init__(self):
        import torch
        from .loading import load_model
        torch.set_num_threads(1);torch.manual_seed(51713)
        self.model,self.tokenizer,self.external=load_model()
    def prepare(self,text,task,preformatted=True):
        import torch
        ids=prompt_ids(self.tokenizer,text,task,preformatted=preformatted)
        return torch.tensor(ids,device=self.model.device,dtype=torch.long)
    def generate(self,prompt,length,sample,task,audit=False):
        import torch
        if length not in (256,512):raise ValueError("Frozen reproduction supports 256/512")
        with torch.no_grad():
            frontier=RelayCounter(cache=True,verify=False)
            engine=Engine(self.model,True,False);runtime=Runtime(self.model,frontier,engine)
            function=mechanism_generator(self.external,frontier,statistics)
            if audit:
                original_reset=frontier.reset
                def reset(*a):original_reset(*a);frontier.actions=[]
                frontier.reset=reset
            capture=OutputCapture(self.tokenizer);text=[None];steps=[0];counts=[0]
            handle=self.model.register_forward_pre_hook(lambda *_:counts.__setitem__(0,counts[0]+1))
            try:
                torch.cuda.synchronize();started=time.perf_counter()
                with engine.installed(),forbid_sdpa(),suppress_official_prints():
                    function(runtime,[prompt],[prompt.numel()],1,text,steps,gen_length=length,block_length=32,
                        threshold=.9,gamma=.8,track_num=4,mask_num=4,verify=False,tokenizer=capture,
                        stop_tokens=sample.get('generation_kwargs',{}).get('until',[]))
                torch.cuda.synchronize();seconds=time.perf_counter()-started
            finally:handle.remove()
            assert counts[0]==len(runtime.calls) and capture.ids is not None and text[0] is not None
            assert not any(c['verify'] for c in runtime.calls)
            row=dict(seconds=seconds,nfe=counts[0],iterations=steps[0],token_ids=capture.ids,raw_decoder_text=text[0],
                ordinary_calls=len(runtime.calls),private_calls=0,ordinary_accepted=frontier.ordinary_accepted,
                actual_ordinary_row_layers=engine.row_layers,optional_row_layers_skipped=engine.optional_skipped_row_layers,
                boundary_bytes=engine.boundary_bytes,phase_counts=engine.phase_counts,
                truncated=126081 not in capture.ids,first_eos=capture.ids.index(126081) if 126081 in capture.ids else None,
                backend='Pinned fused Triton, SDPA forbidden')
            if audit:
                row['actions']=frontier.actions
                row['actions_sha256']=hashlib.sha256(repr(frontier.actions).encode()).hexdigest()
            row['text'],row['output_tokens']=postprocess_output(self.tokenizer,row['token_ids'],sample,task)
            return row

def assert_same_generation(a,b):
    keys=('token_ids','raw_decoder_text','text','nfe','iterations','ordinary_calls','private_calls',
          'ordinary_accepted','actual_ordinary_row_layers','optional_row_layers_skipped','phase_counts','boundary_bytes')
    for key in keys:
        if a[key]!=b[key]:raise AssertionError('Generation changed: '+key)
