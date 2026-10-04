import copy
import io
from contextlib import ExitStack
from unittest.mock import patch
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from relay_cache.dream.generate import Engine, SETTINGS, required_positions, validate_checkpoint, validate_settings, postprocess_output


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return q*cos.unsqueeze(1)+rotate_half(q)*sin.unsqueeze(1), k*cos.unsqueeze(1)+rotate_half(k)*sin.unsqueeze(1)


def repeat_kv(x, groups):
    return x.repeat_interleave(groups, dim=1)


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm=torch.nn.LayerNorm(8)
        self.post_attention_layernorm=torch.nn.LayerNorm(8)
        self.mlp=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.SiLU(),torch.nn.Linear(16,8))
        self.self_attn=torch.nn.Module()
        a=self.self_attn
        a.num_heads=4;a.num_key_value_heads=2;a.num_key_value_groups=2;a.head_dim=2;a.hidden_size=8
        a.q_proj=torch.nn.Linear(8,8);a.k_proj=torch.nn.Linear(8,4);a.v_proj=torch.nn.Linear(8,4);a.o_proj=torch.nn.Linear(8,8,bias=False)

    def forward(self,x,position_embeddings=None,**kwargs):
        # Independent dense full-attention reference, with no cache or stripe scheduling.
        a=self.self_attn;h=self.input_layernorm(x);n=x.shape[1]
        q=a.q_proj(h).view(1,n,4,2).transpose(1,2)
        k=a.k_proj(h).view(1,n,2,2).transpose(1,2)
        v=a.v_proj(h).view(1,n,2,2).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        y=torch.nn.functional.scaled_dot_product_attention(q,repeat_kv(k,2),repeat_kv(v,2),is_causal=False)
        x=x+a.o_proj(y.transpose(1,2).reshape(1,n,8))
        return (x+self.mlp(self.post_attention_layernorm(x)),)


class ToyModel(torch.nn.Module):
    def __init__(self,layers=4):
        super().__init__();self.config=SimpleNamespace(mask_token_id=31)
        self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([ToyLayer() for _ in range(layers)])
        self.emb=torch.nn.Embedding(32,8)

    def forward(self,input_ids,attention_mask='full',**kwargs):
        h=self.emb(input_ids);n=h.shape[1]
        pos=torch.arange(n).view(1,n,1).float();emb=(torch.cos(pos).expand(1,n,2),torch.sin(pos).expand(1,n,2))
        for layer in self.model.layers:h=layer(h,position_embeddings=emb,use_cache=False)[0]
        return h


class DreamTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41);torch.set_num_threads(1)

    def test_shifted_readout_and_dirty_identities_are_mandatory(self):
        old=torch.tensor([[1,2,3,31,31,4]])
        new=torch.tensor([[1,2,9,31,31,4]])
        required,dirty,_=required_positions(new,old,31)
        self.assertEqual(torch.nonzero(required).flatten().tolist(),[2,3,4])
        self.assertTrue(dirty[2]);self.assertFalse(required[5])

    def test_first_mask_and_batch_guards(self):
        x=torch.tensor([[31,2]])
        self.assertTrue(required_positions(x,x,31)[0][0])
        with self.assertRaises(ValueError):required_positions(x.expand(2,-1),None,31)

    def test_full_control_matches_dense_on_changed_real_inputs(self):
        native=ToyModel().eval();adapted=copy.deepcopy(native);engine=Engine(adapted,width=2,full=True)
        with torch.no_grad(),engine.installed():
            for ids in (torch.tensor([[1,2,31,31]]),torch.tensor([[1,2,7,31]])):
                torch.testing.assert_close(adapted(ids),native(ids),rtol=0,atol=0)
        self.assertEqual(engine.skipped,0)

    def test_warm_call_exact_then_history_skips_without_omitting_keys(self):
        model=ToyModel().eval();ids=torch.tensor([[1,2,3,4,5,31]])
        native=model(ids);engine=Engine(model,width=2,budget=2)
        with torch.no_grad(),engine.installed():
            torch.testing.assert_close(model(ids),native,rtol=0,atol=0)
            model(ids)
            self.assertGreater(engine.skipped,0)
            self.assertEqual(engine.required.tolist(),[4,5])
            self.assertTrue(all(k.shape[2]==6 and v.shape[2]==6 for k,v in engine.kv))
            # GQA cache stores two KV heads, not the expanded four attention heads.
            self.assertTrue(all(k.shape[1]==2 for k,v in engine.kv))

    def test_ragged_28_layer_stripes_rotate(self):
        model=ToyModel(layers=28).eval();engine=Engine(model,width=8,budget=2)
        with torch.no_grad(),engine.installed():
            for _ in range(5):model(torch.tensor([[1,2,3,4,31]]))
        self.assertEqual(engine.stripes,4)
        self.assertEqual(engine.phase_counts,[1,1,1,1])

    def test_exception_restores_methods_hooks_and_private_cache(self):
        model=ToyModel().eval();before=[layer.forward for layer in model.model.layers]
        engine=Engine(model)
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            with torch.no_grad(),engine.installed():
                model(torch.tensor([[1,2,31]]));raise RuntimeError('interrupted')
        self.assertEqual([layer.forward for layer in model.model.layers],before)
        self.assertEqual(len(model._forward_pre_hooks),0)
        self.assertIsNone(engine.previous);self.assertTrue(all(item is None for item in engine.kv))

    def test_ar_cache_padding_and_nested_adapters_are_rejected(self):
        model=ToyModel().eval();engine=Engine(model)
        with self.assertRaisesRegex(ValueError,'append-cache'),engine.installed():
            model(torch.tensor([[1,31]]),use_cache=True)
        engine=Engine(model)
        with self.assertRaisesRegex(ValueError,'unpadded'),engine.installed():
            model(torch.tensor([[1,31]]),attention_mask=torch.ones(1,2))

    def test_nested_adapters_and_changed_canvas_are_rejected(self):
        model=ToyModel().eval();engine=Engine(model)
        with torch.no_grad(),engine.installed():
            model(torch.tensor([[1,2,31]]))
            with self.assertRaisesRegex(RuntimeError,'already installed'),Engine(model).installed():pass
            with self.assertRaisesRegex(ValueError,'canvas shape'):
                model(torch.tensor([[1,2,3,31]]))
        self.assertFalse(hasattr(model,'_relay_dream_engine'))

    def test_eos_and_task_stops_do_not_search_for_a_reference(self):
        class Tokenizer:
            eos_token='<EOS>'
            def __call__(self,text):return {'input_ids':[ord(c) for c in text]}
            def decode(self,ids,skip_special_tokens=False):return ''.join(chr(c) for c in ids)
        text,_=postprocess_output(Tokenizer(),'42<EOS>unrelated 999',{'answer':'999'})
        self.assertEqual(text,'42')
        text,_=postprocess_output(Tokenizer(),'42STOP999',{'generation_kwargs':{'until':['STOP']},'answer':'999'})
        self.assertEqual(text,'42')

    def test_model_profile_and_checkpoint_completeness_are_checked(self):
        with self.assertRaises(ValueError):validate_settings({**SETTINGS,'stripe_width':4})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text(json.dumps(dict(model_type='llada')))
            with self.assertRaisesRegex(RuntimeError,'architecture'):validate_checkpoint(root)




class DreamRunnerTests(unittest.TestCase):
    def test_dream_routing_warm_replay_and_native_resume_separation(self):
        from test_reporting import RunnerPresentationRegressionTests
        from dllm_eval import run
        from relay_cache.llada import generate as llada
        helper=RunnerPresentationRegressionTests()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output,logs=Path(directory)/'run',Path(directory)/'log';calls=[]
            args=helper.mocked_environment(stack,output,logs,calls,tasks=['gsm8k'],lengths=[256])
            args.model='dream'
            args.config.write_text(json.dumps(dict(settings=SETTINGS)),encoding='utf8')
            fake=llada.Session
            stack.enter_context(patch('relay_cache.dream.generate.Session',side_effect=lambda backend:fake()))
            run.main()
            manifest=json.loads((output/'manifest.json').read_text(encoding='utf8'))
            self.assertEqual(manifest['model'],SETTINGS['model'])
            self.assertEqual(manifest['revision'],SETTINGS['revision'])
            self.assertEqual(manifest['backend'],'relay')
            self.assertEqual(sum(c[0]=='generate' for c in calls),4)
            self.assertTrue(all('SECRET' not in str(c) for c in calls if c[0]=='prepare'))
            calls.clear();args.resume=True;run.main();self.assertEqual(calls,[])
            args.dream_backend='native'
            with self.assertRaisesRegex(RuntimeError,'manifest changed'):run.main()
            self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
