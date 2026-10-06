import copy
import unittest
from types import SimpleNamespace
import torch
from relay_cache.dream.decoder import Readout,frontier,choose,generate
from relay_cache.dream.generate import Engine,required_positions,backend_settings
from relay_cache.dream.prompts import prompt_ids
from test_dream import ToyModel


class PositionModel(torch.nn.Module):
    """A known position-dependent prediction table, independent of the decoder."""
    def __init__(self,eos_position=None):
        super().__init__()
        self.config=SimpleNamespace(mask_token_id=31,eos_token_id=30)
        self.lm_head=torch.nn.Identity()
        self.eos_position=eos_position
    def forward(self,input_ids,**kwargs):
        n=input_ids.shape[1]
        logits=torch.full((1,n,32),-12.)
        # Row r predicts absolute position r+1, not r.
        for r in range(n):logits[0,r,(r+1)%7+1]=12.
        if self.eos_position is not None:logits[0,self.eos_position-1,30]=24.
        return SimpleNamespace(logits=self.lm_head(logits))


class DecoderTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(41)

    def test_chat_template_is_explicit_and_completion_baselines_unchanged(self):
        class Tokenizer:
            bos_token='<B>'
            def __call__(self,text):self.text=text;return {'input_ids':[1,2]}
            def apply_chat_template(self,messages,**kwargs):self.messages=messages;self.kwargs=kwargs;return [3,4]
        tok=Tokenizer()
        self.assertEqual(prompt_ids(tok,'question',style='chat'),[3,4])
        self.assertEqual(tok.messages,[{'role':'user','content':'question'}])
        self.assertTrue(tok.kwargs['add_generation_prompt'])
        self.assertEqual(prompt_ids(tok,'question'),[1,2]);self.assertEqual(tok.text,'<B>question')
        prompt_ids(tok,'<B>question');self.assertEqual(tok.text,'<B>question')
        with self.assertRaises(ValueError):prompt_ids(tok,'question',style='invented')

    def test_native_and_uncached_sampling_are_not_mislabeled(self):
        native=backend_settings('native');parallel=backend_settings('uncached')
        self.assertEqual((native['alg'],native['temperature'],native['top_p']),('entropy',.1,.9))
        self.assertEqual((parallel['alg'],parallel['block'],parallel['threshold']),('confidence_threshold',32,.9))

    def test_frontier_resolves_holes_before_eos_and_does_not_move_prefix(self):
        ids=torch.tensor([[1,2,31,7,31,30,31,31]])
        active,live,end=frontier(ids,2,31,30,block=1,mask_blocks=4)
        self.assertEqual(active.tolist(),[2]);self.assertEqual(live.tolist(),[2,4]);self.assertEqual(end,5)
        self.assertEqual(ids.tolist(),[[1,2,31,7,31,30,31,31]])

    def test_shifted_decoder_matches_known_dense_prediction_table(self):
        prompt=torch.tensor([[1,2,3]])
        with torch.no_grad():
            selected,sa=generate(PositionModel(),prompt,67,audit=True)
            dense,da=generate(PositionModel(),prompt,67,selected_readout=False,audit=True)
        expected=[p%7+1 for p in range(3,70)]
        self.assertEqual(selected[0,3:].tolist(),expected)
        self.assertTrue(torch.equal(dense,selected));self.assertEqual(sa,da)
        self.assertEqual(sa['calls'],3);self.assertEqual(sa['accepted'],67)

    def test_eos_does_not_leave_unresolved_holes_before_the_response_boundary(self):
        with torch.no_grad():
            ids,stats=generate(PositionModel(eos_position=6),torch.tensor([[1,2]]),64,block=2,audit=True)
        self.assertEqual(stats['calls'],3)
        self.assertFalse(bool(ids[0,2:6].eq(31).any()))
        self.assertEqual(int(ids[0,6]),30)
        self.assertTrue(bool(ids[0,8:].eq(31).all()))

    def test_threshold_inclusive_and_single_fallback(self):
        logits=torch.tensor([[0.,0.],[0.,0.]])
        slots,_,_=choose(logits,.90);self.assertEqual(slots.numel(),1)
        slots,_,_=choose(logits,.50);self.assertEqual(slots.numel(),2)

    def test_readout_cleanup_on_model_exception(self):
        model=PositionModel();before=model.lm_head.forward
        with self.assertRaisesRegex(RuntimeError,'interrupt'):
            with Readout(model).installed():raise RuntimeError('interrupt')
        self.assertEqual(model.lm_head.forward,before)
        self.assertNotIn('forward',model.lm_head.__dict__)

    def test_scheduled_frontier_keeps_live_predecessors_and_changed_identities(self):
        old=torch.tensor([[1,2,31,31,31,31,31,31]])
        new=old.clone();new[0,2]=7
        required,dirty,_=required_positions(new,old,31,torch.tensor([4,5]))
        self.assertEqual(torch.nonzero(required).flatten().tolist(),[2,3,4,5])
        self.assertTrue(dirty[2]);self.assertFalse(required[7])
        with self.assertRaisesRegex(ValueError,'unresolved'):
            required_positions(new,old,31,torch.tensor([2]))

    def test_scheduled_engine_retains_future_kv_and_rotates_only_stable_history(self):
        model=ToyModel().eval();engine=Engine(model,width=2,budget=2)
        ids=torch.tensor([[1,2,3,4,31,31,31,31]])
        with torch.no_grad(),engine.installed():
            engine.schedule(torch.tensor([4,5]));model(ids)
            ids[0,4]=7
            engine.schedule(torch.tensor([5]));model(ids)
            self.assertEqual(engine.required.tolist(),[4,5])
            self.assertTrue(all(int(p)<4 for p in engine.optional))
            self.assertFalse(any(int(p) in (6,7) for p in engine.selected))
            self.assertTrue(all(k.shape[2]==8 for k,v in engine.kv))

    def test_decoded_history_never_refreshes_from_initial_mask_embedding(self):
        model=ToyModel().eval();engine=Engine(model,width=2,budget=8)
        ids=torch.tensor([[1,2,31,31,31,31]])
        with torch.no_grad(),engine.installed():
            engine.schedule(torch.tensor([2,3,4,5]));model(ids)
            ids[0,2]=7
            engine.schedule(torch.tensor([4,5]));model(ids)
            expected=model.emb(ids)[0,2]
            torch.testing.assert_close(engine.boundaries[0][0,2],expected,rtol=0,atol=0)
            model(ids)  # Other stripe refreshes the now-stable decoded history.
            model(ids)  # First stripe must read the decoded embedding, not MASK.
            self.assertEqual(engine.phase,0);self.assertIn(2,engine.optional.tolist())
            torch.testing.assert_close(engine.boundaries[0][0,2],expected,rtol=0,atol=0)
            layer=model.model.layers[0]
            normalized=layer.input_layernorm(model.emb(ids)[:,2:3])
            expected_v=layer.self_attn.v_proj(normalized).view(1,1,2,2).transpose(1,2)
            torch.testing.assert_close(engine.kv[0][1][:,:,2:3],expected_v,rtol=1e-6,atol=1e-7)


if __name__=='__main__':unittest.main()
