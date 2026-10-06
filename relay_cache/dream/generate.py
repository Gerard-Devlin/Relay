"""DREAM Instruct integration and shift-aware Relay stripe caching.

The native backend retains the official entropy sampler. Relay uses an adaptive
32-token sliding frontier. Live MASK queries, their shifted readout
predecessors, and changed identities traverse every layer. Stable history uses
rotating stripes; it still contributes cached KV to every attention call.
This adapter is approximate and has no DREAM performance/quality claim yet.
"""
import importlib
import json
import math
import time
from contextlib import contextmanager, nullcontext
from .loading import MODEL_ID, REVISION, load_model, validate_checkpoint

SETTINGS = dict(method="relay_dream", model=MODEL_ID, revision=REVISION,
    stripe_width=8, history_budget=128, lengths=[256, 512], seed=51713,
    precision="BF16", alg="confidence_threshold", temperature=0.0,
    top_p=None, top_k=None, block=32, threshold=.90, track=4, mask=4,
    verify=False, schedule="Shift-aware Relay sliding frontier; adaptive calls",
    selected_readout=True)
from .prompts import CHAT_POLICY
SETTINGS['prompt_policy'] = CHAT_POLICY


def backend_settings(backend):
    if backend not in ('relay','native','uncached'):
        raise ValueError('Unknown DREAM backend')
    value = dict(SETTINGS,backend=backend)
    if backend == 'native':
        value.update(alg='entropy',temperature=.1,top_p=.9,alg_temp=0.,eps=.001,
                     steps={'256':256,'512':512},selected_readout=False,
                     schedule='Official DREAM Instruct full-canvas entropy sampler')
    return value


def validate_settings(value):
    if value != SETTINGS:
        raise ValueError("DREAM reproduction settings changed")


def required_positions(ids, previous, mask_id, live_positions=None):
    """DREAM shifts logits right: predicting position i requires query i-1."""
    import torch
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("DREAM Relay requires a nonempty, unpadded batch of one")
    masked = ids[0] == mask_id
    if live_positions is None:
        live_positions = torch.nonzero(masked, as_tuple=False).flatten()
    if live_positions.ndim != 1 or bool((live_positions<0).any()) or bool((live_positions>=ids.shape[1]).any()):
        raise ValueError('Invalid live DREAM positions')
    if not bool(masked[live_positions].all()):
        raise ValueError('Live DREAM frontier must contain unresolved positions')
    required = torch.zeros_like(masked)
    required[live_positions] = True
    readout = live_positions.sub(1).clamp_min(0)
    required[readout] = True
    dirty = torch.ones_like(required) if previous is None else ids[0] != previous[0]
    return required | dirty, dirty, masked


class Engine:
    """Instance-local layer adapter; temporary state never enters native AR cache."""
    def __init__(self, model, width=8, budget=128, full=False):
        if width <= 0 or budget < 0:
            raise ValueError("Invalid DREAM stripe policy")
        if model.training:
            raise ValueError("DREAM Relay requires model.eval()")
        self.model, self.layers = model, model.model.layers
        self.width, self.budget, self.full = width, budget, full
        self.stripes = math.ceil(len(self.layers) / width)
        self.module = importlib.import_module(type(self.layers[0]).__module__)
        self.mask_id = model.config.mask_token_id
        self.previous = self.boundaries = self.saliency = None
        self.kv = [None] * len(self.layers)
        self.calls = self.row_layers = self.skipped = 0
        self.phase_counts = [0] * self.stripes
        self.active = False
        self.live_positions = None

    def schedule(self, live_positions):
        self.live_positions = live_positions

    def begin(self, _model, args, kwargs):
        import torch
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or kwargs.get("past_key_values") is not None or kwargs.get("use_cache", False):
            raise ValueError("Use token IDs and disable autoregressive append-cache for DREAM")
        if self.previous is not None and ids.shape != self.previous.shape:
            raise ValueError("DREAM canvas shape changed within a request")
        required, dirty, masked = required_positions(ids, self.previous, self.mask_id,self.live_positions)
        mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        if isinstance(mask, torch.Tensor):
            raise ValueError("DREAM Relay currently requires an unpadded full-attention canvas")
        self.first = self.previous is None
        self.phase = max(0, self.calls - 1) % self.stripes
        self.required = torch.nonzero(required, as_tuple=False).flatten()
        # History stripes only apply to immutable decoded/prompt identities.
        # Distant future MASK KV remains cached until it enters the live frontier.
        eligible = torch.nonzero(~required & ~masked, as_tuple=False).flatten()
        spare = max(0, self.budget - int((dirty & ~masked).sum()))
        if eligible.numel() and spare:
            score = self.saliency[eligible] if self.saliency is not None else eligible.float()
            order = torch.argsort(score, stable=True)
            self.optional = eligible[order[-spare:]].sort().values
        else:
            self.optional = eligible[:0]
        self.selected = torch.cat((self.required, self.optional)).sort().values
        self.previous = ids.clone()
        self.calls += 1
        if not self.first:
            self.phase_counts[self.phase] += 1
        self.seen = []

    def forward(self, layer_index, hidden_states, **kwargs):
        import torch
        layer = self.layers[layer_index]
        if layer_index != len(self.seen):
            raise RuntimeError("DREAM layer order changed")
        self.seen.append(layer_index)
        if kwargs.get("past_key_value") is not None or kwargs.get("use_cache") or kwargs.get("output_attentions"):
            raise ValueError("DREAM Relay does not support AR caches or attention-output requests")
        if self.boundaries is None:
            self.boundaries = torch.empty((self.stripes + 1, *hidden_states.shape),
                                          device=hidden_states.device, dtype=hidden_states.dtype)
        if layer_index == 0:
            # The model already computed current embeddings for every identity.
            # A newly decoded token must never later refresh from its old MASK.
            self.boundaries[0].copy_(hidden_states)
        stripe = layer_index // self.width
        if layer_index % self.width == 0:
            if self.first:
                self.boundaries[stripe].copy_(hidden_states)
            elif not self.full and stripe == self.phase and self.optional.numel():
                hidden_states = hidden_states.clone()
                hidden_states.index_copy_(1, self.optional, self.boundaries[stripe].index_select(1, self.optional))
        rows = torch.arange(hidden_states.shape[1], device=hidden_states.device) if self.first or self.full else (
            self.selected if stripe == self.phase else self.required)
        self.row_layers += rows.numel()
        self.skipped += hidden_states.shape[1] - rows.numel()
        if rows.numel():
            x = hidden_states.index_select(1, rows)
            residual = x
            x = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_proj(x).view(1, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(1, -1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(1, -1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            embeddings = kwargs.get("position_embeddings")
            if embeddings is None:
                raise ValueError("DREAM shared absolute RoPE embeddings required")
            cos, sin = (part.index_select(1, rows) for part in embeddings)
            q, k = self.module.apply_rotary_pos_emb(q, k, cos, sin)
            if self.kv[layer_index] is None:
                if not self.first or rows.numel() != hidden_states.shape[1]:
                    raise RuntimeError("DREAM cache must initialize every position")
                self.kv[layer_index] = (k.clone(), v.clone())
            else:
                self.kv[layer_index][0].index_copy_(2, rows, k)
                self.kv[layer_index][1].index_copy_(2, rows, v)
            all_k, all_v = (self.module.repeat_kv(part, attn.num_key_value_groups) for part in self.kv[layer_index])
            # Same full-attention SDPA operation as the pinned official DREAM model.
            y = torch.nn.functional.scaled_dot_product_attention(q.contiguous(), all_k.contiguous(),
                all_v.contiguous(), attn_mask=None, dropout_p=0.0, is_causal=False)
            y = residual + attn.o_proj(y.transpose(1, 2).contiguous().view(1, rows.numel(), attn.hidden_size))
            y = y + layer.mlp(layer.post_attention_layernorm(y))
            output = hidden_states.clone()
            output.index_copy_(1, rows, y)
            if layer_index == min(3, len(self.layers) - 1):
                # Raw QK importance, with no N-by-N tracking matrix materialization.
                self.saliency = torch.einsum('hd,hnd->n', q[0].float().mean(1), all_k[0].float()) / attn.num_heads
        else:
            output = hidden_states
        if (layer_index + 1) % self.width == 0 or layer_index + 1 == len(self.layers):
            self.boundaries[stripe + 1].index_copy_(1, rows, output.index_select(1, rows))
        return (output,)

    @contextmanager
    def installed(self):
        if self.active or getattr(self.model, "_relay_dream_engine", None) is not None:
            raise RuntimeError("DREAM adapter already installed")
        self.active = True
        self.model._relay_dream_engine = self
        saved = [("forward" in layer.__dict__, layer.__dict__.get("forward")) for layer in self.layers]
        handle = None
        try:
            for index, layer in enumerate(self.layers):
                def forward(hidden_states, _index=index, **kwargs):
                    return self.forward(_index, hidden_states, **kwargs)
                layer.forward = forward
            handle = self.model.register_forward_pre_hook(self.begin, with_kwargs=True)
            yield self
            if self.seen != list(range(len(self.layers))):
                raise RuntimeError("Incomplete DREAM forward")
        finally:
            if handle is not None:
                handle.remove()
            for layer, (had_attribute, value) in zip(self.layers, saved):
                if had_attribute:
                    layer.forward = value
                else:
                    del layer.forward
            self.active = False
            del self.model._relay_dream_engine
            self.kv = [None] * len(self.layers)
            self.boundaries = self.previous = self.saliency = None
            self.live_positions = None


def postprocess_output(tokenizer, raw, sample):
    # Match DREAM's benchmark response boundary before the common task scorer.
    text = raw.split(tokenizer.eos_token, 1)[0] if tokenizer.eos_token else raw
    for stop in sample.get("generation_kwargs", {}).get("until", []):
        text = text.split(stop, 1)[0]
    processed = tokenizer(text)["input_ids"]
    return tokenizer.decode(processed, skip_special_tokens=True), processed


def assert_same_generation(a, b):
    for key in ("token_ids", "text", "raw_decoder_text", "nfe", "iterations", "phase_counts",
                "actual_ordinary_row_layers", "optional_row_layers_skipped"):
        if a[key] != b[key]:
            raise AssertionError("DREAM warm replay changed: " + key)


class Session:
    def __init__(self, backend="relay", model=None, tokenizer=None):
        import torch
        if backend not in ("relay", "native", "uncached"):
            raise ValueError("Unknown DREAM backend")
        torch.set_num_threads(1)
        self.model, self.tokenizer = load_model() if model is None else (model, tokenizer)
        self.model.config.use_cache = False
        self.backend = backend
        self.settings = backend_settings(backend)

    def prepare(self, text, task, preformatted=True):
        import torch
        from .prompts import prompt_ids
        ids = prompt_ids(self.tokenizer, text, style='chat')
        return torch.tensor([ids], device=self.model.device, dtype=torch.long)

    def generate(self, prompt, length, sample, task, audit=False):
        import torch
        if length not in SETTINGS["lengths"]:
            raise ValueError("DREAM profile supports 256/512")
        engine = Engine(self.model) if self.backend == "relay" else None
        calls, actions, previous = [0], [], [None]
        def count(*_):
            calls[0] += 1
        def observe(step, x, _logits):
            if audit and previous[0] is not None:
                positions = torch.nonzero(x[0] != previous[0][0], as_tuple=False).flatten()
                actions.append((positions.tolist(), x[0, positions].tolist()))
            if audit:
                previous[0] = x.clone()
            return x
        handle = self.model.register_forward_pre_hook(count)
        devices = [self.model.device.index or 0] if self.model.device.type == "cuda" else []
        synchronize = torch.cuda.synchronize if devices else lambda: None
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                torch.manual_seed(SETTINGS["seed"])
                synchronize(); started = time.perf_counter()
                if self.backend == 'native':
                    p=self.settings
                    sequence = self.model.diffusion_generate(prompt,max_new_tokens=length,
                        steps=p['steps'][str(length)],alg=p['alg'],alg_temp=p['alg_temp'],
                        temperature=p['temperature'],top_p=p['top_p'],top_k=p['top_k'],eps=p['eps'],
                        mask_token_id=self.model.config.mask_token_id,generation_tokens_hook_func=observe)
                else:
                    from .decoder import generate
                    sequence,decoding=generate(self.model,prompt,length,engine=engine,
                        block=SETTINGS['block'],mask_blocks=SETTINGS['mask'],threshold=SETTINGS['threshold'],
                        selected_readout=SETTINGS['selected_readout'],audit=audit)
                    if audit: actions=decoding['actions']
                synchronize()
                seconds = time.perf_counter() - started
                ids = sequence[0, prompt.shape[1]:].tolist()
                raw = self.tokenizer.decode(ids, skip_special_tokens=False)
        finally:
            handle.remove()
        text, processed = postprocess_output(self.tokenizer, raw, sample)
        eos = self.tokenizer.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        first_eos = next((i for i, v in enumerate(ids) if v in eos), None)
        row = dict(seconds=seconds, token_ids=ids, raw_decoder_text=raw, text=text,
            output_tokens=sum(v not in eos for v in processed), nfe=calls[0], iterations=calls[0],
            ordinary_calls=calls[0], private_calls=0, backend="DREAM official SDPA; " + self.backend,
            model=MODEL_ID, revision=REVISION, first_eos=first_eos, truncated=first_eos is None,
            sampling=self.settings,
            actual_ordinary_row_layers=engine.row_layers if engine else calls[0]*sequence.numel()*len(self.model.model.layers),
            optional_row_layers_skipped=engine.skipped if engine else 0,
            phase_counts=engine.phase_counts if engine else [])
        if audit:
            row["actions"] = actions
        return row
