"""Shift-aware, confidence-parallel Relay decoding for DREAM.

Uses a sliding frontier of unresolved token positions. Unlike LLaDA, a token
at absolute position i is read from hidden row i-1. No speculative verification
or reference answer is involved. This sampler is approximate, not equivalent
to DREAM's fixed-step entropy schedule.
"""
from contextlib import contextmanager


class Readout:
    """Project consumed hidden rows; both cached and uncached paths use this head."""
    def __init__(self, model, selected=True):
        self.model, self.selected, self.positions = model, selected, None

    def __call__(self, ids, targets):
        self.positions = (targets-1).clamp_min(0)
        logits = self.model(input_ids=ids, attention_mask='full', use_cache=False).logits
        return logits[0] if self.selected else logits[0].index_select(0, self.positions)

    @contextmanager
    def installed(self):
        if not self.selected:
            yield self
            return
        head = self.model.lm_head
        had_attribute, saved = 'forward' in head.__dict__, head.__dict__.get('forward')
        original = head.forward
        def forward(hidden):
            if self.positions is None:
                raise RuntimeError('Readout positions were not scheduled')
            return original(hidden.index_select(1, self.positions))
        head.forward = forward
        try:
            yield self
        finally:
            if had_attribute: head.forward = saved
            else: del head.forward
            self.positions = None


def frontier(ids, prompt_length, mask_id, eos_id, block=32, mask_blocks=4):
    """EOS defines the response boundary; all holes before it must be resolved."""
    import torch
    generated = ids[0, prompt_length:]
    eos = torch.nonzero(generated == eos_id, as_tuple=False).flatten()
    end = prompt_length + (int(eos[0]) if eos.numel() else generated.numel())
    positions = torch.nonzero(ids[0, prompt_length:end] == mask_id, as_tuple=False).flatten()+prompt_length
    return positions[:block], positions[:block*mask_blocks], end


def choose(logits, threshold=.90):
    """Native >= threshold rule plus one highest-confidence fallback."""
    import torch
    # DREAM's official greedy sampler computes softmax in the logits dtype.
    probability = torch.softmax(logits, dim=-1)
    confidence, tokens = probability.max(dim=-1)
    order = confidence.argsort(descending=True)
    accepted = order[confidence[order] >= threshold]
    if not accepted.numel(): accepted = order[:1]
    return accepted, tokens, confidence


def generate(model, prompt, length, *, engine=None, block=32, mask_blocks=4,
             threshold=.90, selected_readout=True, audit=False):
    import torch
    from contextlib import nullcontext
    if prompt.ndim != 2 or prompt.shape[0] != 1 or not prompt.shape[1] or length<=0:
        raise ValueError('DREAM parallel decoder needs one nonempty unpadded prompt')
    mask_id, eos_id = model.config.mask_token_id, model.config.eos_token_id
    if not isinstance(eos_id, int):
        raise ValueError('DREAM single EOS identity required')
    if bool(prompt.eq(mask_id).any()):
        raise ValueError('A legitimate prompt must not contain MASK tokens')
    ids = torch.cat((prompt,torch.full((1,length),mask_id,device=prompt.device,dtype=prompt.dtype)),dim=1)
    readout = Readout(model,selected=selected_readout)
    actions, calls, accepted_total = [], 0, 0
    with readout.installed(), engine.installed() if engine else nullcontext():
        for _ in range(length):
            active, live, _ = frontier(ids,prompt.shape[1],mask_id,eos_id,block,mask_blocks)
            if not active.numel(): break
            if engine is not None: engine.schedule(live)
            logits = readout(ids,active)
            slots,tokens,confidence = choose(logits,threshold)
            positions,values = active[slots],tokens[slots]
            if bool(values.eq(mask_id).any()):
                raise RuntimeError('DREAM proposed MASK; decoder cannot make legal progress')
            ids[0,positions] = values
            calls += 1; accepted_total += positions.numel()
            if audit:
                actions.append(dict(positions=positions.tolist(),tokens=values.tolist(),
                                    confidence=confidence[slots].float().tolist()))
        else:
            active,_,_ = frontier(ids,prompt.shape[1],mask_id,eos_id,block,mask_blocks)
            if active.numel(): raise RuntimeError('DREAM parallel decoding exhausted its progress bound')
    return ids,dict(calls=calls,accepted=accepted_total,actions=actions)
