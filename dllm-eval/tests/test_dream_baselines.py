import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from dllm_eval.dream_baselines import settings,generate_official,official_prompt
from relay_cache.dream.prompts import prompt_ids
from relay_cache.dream.loading import validate_checkpoint, load_checkpoint


class DreamBaselineTests(unittest.TestCase):
    def test_dream_registry_does_not_expand_historical_llada_campaign(self):
        from dllm_eval.baseline import SOURCES,DREAM_SOURCES
        self.assertEqual(set(SOURCES),{'d2cache','elastic_cache'})
        self.assertIn('fast_dllm_v1',DREAM_SOURCES)
    def test_profiles_keep_official_distinctions(self):
        p=[settings(m) for m in ('fast_dllm_v1','d2cache','elastic_cache')]
        self.assertEqual(len({v['model'] for v in p}),1)
        self.assertTrue(all(v['parallel_decoding'] and v['batch_size']==1 for v in p))
        self.assertFalse(p[0]['dual_cache']);self.assertTrue(p[0]['use_cache'])
        self.assertEqual(p[1]['generation_sigma'],0.);self.assertEqual(p[1]['inflate_w'],4)
        self.assertEqual(p[1]['top_p'],.9)
        self.assertEqual(p[2]['window_length'],32);self.assertTrue(p[2]['stop_until_eos'])
        with self.assertRaises(ValueError):settings('unknown')

    def test_prompt_is_prepared_text_plus_official_bos(self):
        class Tokenizer:
            bos_token='<bos>'
            def __call__(self,s):self.text=s;return {'input_ids':[3,4]}
            def apply_chat_template(self,*args,**kwargs):raise AssertionError('Wrong model template')
        tokenizer=Tokenizer()
        self.assertEqual(prompt_ids(tokenizer,'QUESTION'),[3,4]);self.assertEqual(tokenizer.text,'<bos>QUESTION')
        with self.assertRaises(ValueError):prompt_ids(tokenizer,'')

    def invoke(self,method):
        output=torch.arange(35).view(1,-1)
        model=SimpleNamespace(diffusion_generate=Mock(return_value=SimpleNamespace(sequences=output)))
        session=SimpleNamespace(method=method,model=model,settings=settings(method),
            tokenizer=SimpleNamespace(eos_token_id=151643,bos_token_id=151643))
        prompt={'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones(1,3,dtype=torch.bool)}
        result=generate_official(session,prompt,32)
        self.assertEqual(result.tolist(),[list(range(3,35))])
        return model.diffusion_generate.call_args.kwargs

    def test_v1_invocation_binds_prefix_parallel(self):
        args=self.invoke('fast_dllm_v1')
        self.assertEqual(args['steps'],1);self.assertEqual(args['block_length'],32)
        self.assertFalse(args['dual_cache']);self.assertEqual(args['alg'],'confidence_threshold')
        self.assertNotIn('gamma',args)

    def test_v1_kernel_variants_keep_official_sampler_arguments(self):
        for method in ('fast_dllm_v1_flash','fast_dllm_v1_no_flash'):
            args=self.invoke(method)
            self.assertEqual(args['block_length'],32);self.assertEqual(args['threshold'],.90)
            self.assertFalse(args['dual_cache']);self.assertEqual(args['alg'],'confidence_threshold')
            self.assertNotIn('gamma',args)

    def test_elastic_invocation_preserves_stop_and_tracking(self):
        args=self.invoke('elastic_cache')
        self.assertEqual(args['window_length'],32);self.assertEqual(args['track_num'],1)
        self.assertEqual(args['eos_id'],151643);self.assertTrue(args['block_caching'])

    def test_d2_invocation_uses_shift_aware_official_generator(self):
        gen=Mock(return_value=[SimpleNamespace(generated_tokens=torch.ones(1,32,dtype=torch.long))])
        s=SimpleNamespace(method='d2cache',settings=settings('d2cache'),generator=gen,model=object(),cache_cls=object())
        prompt=dict(input_ids=torch.ones(1,2,dtype=torch.long),attention_mask=torch.ones(1,2))
        self.assertEqual(generate_official(s,prompt,32).shape,(1,32))
        args=gen.call_args.kwargs
        self.assertEqual(args['mask_token_id'],151666);self.assertEqual(args['sigma'],0.)
        self.assertEqual(args['threshold'],.90);self.assertFalse(args['stop_until_eos'])
        self.assertEqual(args['top_p'],.9)

    def test_elastic_native_humaneval_script_parameters(self):
        p=settings('elastic_cache','humaneval')
        self.assertEqual((p['window_length'],p['gamma']),(16,.98))
        output=torch.ones(1,259,dtype=torch.long)
        model=SimpleNamespace(diffusion_generate=Mock(return_value=SimpleNamespace(sequences=output)))
        session=SimpleNamespace(method='elastic_cache',model=model,tokenizer=SimpleNamespace(eos_token_id=151643,bos_token_id=151665))
        prompt=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3),task='humaneval')
        generate_official(session,prompt,256)
        args=model.diffusion_generate.call_args.kwargs
        self.assertEqual((args['window_length'],args['steps'],args['gamma']),(16,16,.98))

    def test_d2_native_instruct_task_has_only_legitimate_code_prompt(self):
        text='def add(a, b):\n    """Return the sum."""'
        result=official_prompt('d2cache',text,'humaneval')
        self.assertEqual(result,'Write a solution to the following problem and make sure that it passes the tests:\n```python\n'+text+'\n```\nHere is the completed function:\n```python\n'+text+'\n')
        self.assertEqual(official_prompt('elastic_cache',text,'humaneval'),text)
        self.assertEqual(official_prompt('d2cache','QUESTION','gsm8k'),'QUESTION')

    def checkpoint(self,root,name='model.safetensors'):
        root.mkdir(parents=True,exist_ok=True)
        config=dict(model_type='Dream',num_hidden_layers=28,hidden_size=3584,num_attention_heads=28,
                    num_key_value_heads=4,mask_token_id=151666)
        (root/'config.json').write_text(json.dumps(config))
        for file in ('configuration_dream.py','modeling_dream.py','generation_utils.py','tokenization_dream.py',
                     'tokenizer_config.json','vocab.json','merges.txt'):(root/file).write_text('test')
        (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':name}}))

    @unittest.skipIf(os.name == 'nt', 'HF symlink lifecycle is checked on the Linux server')
    def test_hf_standard_blob_link_allowed_but_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            repo=Path(t)/'models--Dream-org--Dream-v0-Instruct-7B';root=repo/'snapshots'/'revision'
            self.checkpoint(root);(repo/'blobs').mkdir();blob=repo/'blobs'/'sha';blob.write_bytes(b'weights')
            shard=root/'model.safetensors';shard.symlink_to(blob)
            self.assertEqual(validate_checkpoint(root),root.resolve())
            shard.unlink();outside=Path(t)/'outside';outside.write_bytes(b'wrong');shard.symlink_to(outside)
            with self.assertRaises(RuntimeError):validate_checkpoint(root)

    def test_checkpoint_index_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'snapshot';self.checkpoint(root,'../model.safetensors')
            (root.parent/'model.safetensors').write_bytes(b'weights')
            with self.assertRaisesRegex(RuntimeError,'shard name'):validate_checkpoint(root)

    def test_loading_info_does_not_enter_broken_official_wrapper(self):
        from transformers import PreTrainedModel
        cls=type('OfficialDream',(),{});cls.__module__='official_dream.modeling_dream'
        model=SimpleNamespace();config=object()
        generation=SimpleNamespace(DreamGenerationConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)))
        def loader(actual,*args,**kwargs):
            self.assertIs(actual,cls);self.assertTrue(kwargs['output_loading_info'])
            self.assertTrue(kwargs['local_files_only']);return model,{}
        with patch.object(PreTrainedModel,'from_pretrained',classmethod(loader)),patch('importlib.import_module',return_value=generation):
            self.assertIs(load_checkpoint(cls,'checkpoint','eager'),model)
        self.assertIs(model.generation_config,config)

    def test_loading_mismatch_still_rejected(self):
        from transformers import PreTrainedModel
        cls=type('OfficialDream',(),{});cls.__module__='official_dream.modeling_dream'
        with patch.object(PreTrainedModel,'from_pretrained',classmethod(lambda *a,**k:(object(),{'missing_keys':['weight']}))):
            with self.assertRaisesRegex(RuntimeError,'mismatch'):load_checkpoint(cls,'checkpoint')

if __name__=='__main__':unittest.main()
