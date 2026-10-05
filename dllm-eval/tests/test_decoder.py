"""CPU coverage of the real ordinary decoder and its compact cache adapter."""
import ast
import contextlib
import importlib
import io
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.nn.functional as F
from relay_cache import cache, execution


def load_decoder(source):
    tree = ast.parse(source.read_text())
    names = {'get_rotary_embedding', 'make_blocks', 'generate_with_Flash_dLLM'}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, F=F, einsum=torch.einsum, time=time, __name__=__name__)
    exec(compile(tree, str(source), 'exec'), scope)
    return scope['generate_with_Flash_dLLM']


def flash_fused_elastic_cache(block, x, layer, positions, lengths, softmax_scale=None):
    block.k_cache.index_copy_(0, positions[0].long(), x)
    block.v_cache.index_copy_(0, positions[0].long(), x * .5)
    return (x + (layer + 1) / 128).unsqueeze(0)


class Block:
    pass


class Predictor(torch.nn.Module):
    """Deterministic CPU predictor; all model/cache calls use the real decoder."""
    def __init__(self, prompt_length, length, *, eos=None, fallback=False, fail_at=None):
        super().__init__()
        self.config = NS(d_model=16, n_layers=32, n_heads=1, rope_theta=10000., vocab_size=16)
        self.model = NS(transformer=NS(blocks=[Block() for _ in range(32)], ln_f=torch.nn.Identity()))
        self.device = torch.device('cpu')
        self.dtype = torch.float64
        self.prompt_length, self.length, self.eos = prompt_length, length, eos
        self.fallback, self.fail_at = fallback, fail_at
        self.calls = []

    def forward(self, ids, *, positions, lengths, use_cache):
        if lengths[-1]:
            raise AssertionError('Unexpected private model call')
        self.calls.append((ids.clone(), positions[0].clone(), lengths[2].clone(), lengths[3].clone()))
        if self.fail_at == len(self.calls):
            raise RuntimeError('Injected model failure')
        q = positions[0]
        x = ids[0, :, None].double().expand(-1, 16).clone()
        module = importlib.import_module(type(self.model.transformer.blocks[0]).__module__)
        for layer, block in enumerate(self.model.transformer.blocks):
            x = module.flash_fused_elastic_cache(block, x, layer, positions, lengths).squeeze(0)
        positions[4].add_(torch.arange(positions[4].numel()) % 11)
        logits = torch.zeros((q.numel(), 16), dtype=self.dtype)
        for row, absolute in enumerate(q.tolist()):
            position = absolute - self.prompt_length
            token = 2 + (absolute * 3) % 14
            if position >= self.length or (self.eos is not None and position == self.eos):
                token = 1
            probability = .4 + (absolute % 5) * .04 if self.fallback else (.96 if (absolute + len(self.calls)) % 3 else .60)
            logits[row, token] = math.log(15 * probability / (1 - probability))
        return NS(logits=self.model.transformer.ln_f(logits.unsqueeze(0)))


class Tokenizer:
    def __init__(self):
        self.raw = []

    def decode(self, ids, skip_special_tokens=False):
        values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        if not skip_special_tokens:
            self.raw.append(values)
        return ' '.join(map(str, (v for v in values if not skip_special_tokens or v != 1)))

    def __call__(self, text):
        return {'input_ids': [int(v) for v in text.split()]}


def exercise(source, execution_module=execution, cache_module=cache, *, length=64, eos=None,
             fallback=False, fail_at=None, track=4, stop_tokens=()):
    decoder = load_decoder(source)
    model = Predictor(5, length, eos=eos, fallback=fallback, fail_at=fail_at)
    actions = []

    class Frontier(cache_module.RelayFrontier):
        def commit(self, positions, values):
            super().commit(positions, values)
            actions.append((tuple(positions.tolist()), tuple(values.tolist())))

    frontier = Frontier(cache=True)
    engine = cache_module.Engine(model, True, False)
    runtime = cache_module.Runtime(model, frontier, engine)
    function = execution_module.mechanism_generator(decoder, frontier, execution_module.statistics)
    tokenizer = Tokenizer()
    response, steps = [None], [0]
    prompt = torch.tensor([7, 8, 9, 10, 11])
    try:
        with engine.installed(), contextlib.redirect_stdout(io.StringIO()), patch('torch.cuda.max_memory_allocated', return_value=0):
            function(runtime, [prompt], [5], 1, response, steps, gen_length=length, block_length=32,
                     mask_id=0, eos_id=1, threshold=.9, track_num=track, mask_num=4,
                     tokenizer=tokenizer, stop_tokens=list(stop_tokens))
    except BaseException:
        assert not model.model.transformer.ln_f._forward_hooks
        assert engine.call is None
        assert model.calls and '_scope_frontier' not in decoder.__wrapped__.__globals__
        assert flash_fused_elastic_cache is engine.original
        raise
    initialized = engine.labels >= 0
    return dict(response=response, steps=steps, actions=actions, raw=tokenizer.raw, calls=model.calls,
                kv=[(b.k_cache[initialized].clone(), b.v_cache[initialized].clone()) for b in model.model.transformer.blocks],
                boundaries=engine.boundaries[:, initialized].clone(), labels=engine.labels.clone(), dirty=frontier.dirty.clone(),
                plans=frontier.plans, row_layers=engine.row_layers, skipped=engine.optional_skipped_row_layers,
                phases=engine.phase_counts, nfe=len(runtime.calls), hooks=len(model.model.transformer.ln_f._forward_hooks))


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source = Path(execution.__file__).parent/'llada/model/decoder.py'

    def test_real_decoder_one_model_call_per_iteration(self):
        result = exercise(self.source, length=256)
        self.assertEqual(result['nfe'], result['steps'][0])
        self.assertEqual(result['nfe'], len(result['actions']))
        self.assertEqual(result['hooks'], 0)
        self.assertGreater(result['skipped'], 0)
        self.assertTrue(all(result['phases']))

    def test_native_argmax_fallback_still_commits_one(self):
        result = exercise(self.source, length=32, fallback=True)
        self.assertTrue(all(len(positions) == 1 for positions, _ in result['actions']))
        self.assertGreaterEqual(result['nfe'], 32)
        self.assertEqual(result['nfe'], result['steps'][0])

    def test_eos_keeps_existing_stop_behavior(self):
        result = exercise(self.source, length=64, eos=9)
        self.assertIn(1, result['raw'][0])
        self.assertLess(result['nfe'], 64)

    def test_compact_ragged_readout_keeps_full_normalization(self):
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = NS(transformer=NS(ln_f=torch.nn.LayerNorm(8, dtype=torch.float64)))

            def forward(self, hidden):
                return NS(logits=self.model.transformer.ln_f(hidden))

        model = Head()
        hidden = torch.arange(80).double().reshape(1, 10, 8)
        expected = model(hidden).logits[:, 2:7]
        actual = execution.ModelReadout(model, compact=True)(hidden, readout_rows=(2, 5))
        torch.testing.assert_close(actual.logits[2:7], expected.squeeze(0), rtol=0, atol=0)
        self.assertEqual(len(model.model.transformer.ln_f._forward_hooks), 0)

    def test_exception_restores_cache_and_readout_hooks(self):
        # Preserve the existing incomplete-layer assertion and its original cause.
        with self.assertRaises(AssertionError) as caught:
            exercise(self.source, fail_at=2)
        self.assertIsInstance(caught.exception.__context__, RuntimeError)
        self.assertIn('Injected model failure', str(caught.exception.__context__))

    def test_disabled_feature_cannot_be_reenabled(self):
        function = load_decoder(self.source)
        with self.assertRaises(TypeError):
            function(None, [], [], 1, [], [], verify=True)

    def test_adapter_rejects_additional_model_calls(self):
        with self.assertRaisesRegex(ValueError, 'one ordinary model call'):
            execution.generator(extra_calls)


def extra_calls(model):
    model()
    model()


if __name__ == '__main__':
    unittest.main()
