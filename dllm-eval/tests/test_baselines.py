import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from dllm_eval.baseline import settings, source_manifest, verify_upstream, assert_same_generation, SOURCES, configure_sparse_kernel, SPARSE_KERNEL_OPTIONS
from relay_cache.utils import sha256, generation_prompt


class BaselineTests(unittest.TestCase):
    def test_official_es_profile(self):
        value = settings('es_dllm')
        self.assertFalse(value['parallel_mode'])
        self.assertEqual(value['per_task']['math'], [256,256,8])
        self.assertEqual(value['per_task']['gsm8k'], [64,64,16])
        self.assertEqual(value['proportion_steps'], [[1,0],[.5,.125],[.25,.25]])

    def test_short_context_sparse_profile(self):
        value = settings('sparsed')
        self.assertEqual((value['skip'],value['select'],value['attention_block_size']),(.2,.5,32))

    def test_sparse_kernel_preserves_inputs_and_mask(self):
        observed=[];q,k,v,mask=object(),object(),object(),object();result=object()
        def original(*args, **kwargs):
            observed.append((args,kwargs));return result
        model=SimpleNamespace(flex_attn=original)
        configure_sparse_kernel(model)
        self.assertIs(model.flex_attn(q,k,v,block_mask=mask),result)
        args,kwargs=observed[0]
        self.assertEqual(args,(q,k,v));self.assertIs(kwargs['block_mask'],mask)
        self.assertEqual(kwargs['kernel_options'],SPARSE_KERNEL_OPTIONS)
        with self.assertRaises(RuntimeError):model.flex_attn(q,k,v,kernel_options={})

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):settings('flash')

    def test_upstream_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.py';p.write_text('original')
            manifest={'files':{'model.py':sha256(p)}}
            verify_upstream(tmp,manifest)
            p.write_text('changed')
            with self.assertRaises(RuntimeError):verify_upstream(tmp,manifest)

    def test_upstream_new_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'new.py').write_text('unexpected')
            with self.assertRaises(RuntimeError):verify_upstream(tmp,{'files':{}})

    def test_wrong_upstream_revision_rejected(self):
        with patch('dllm_eval.baseline.subprocess.check_output',return_value='wrong'):
            with self.assertRaises(RuntimeError):source_manifest('es_dllm','.')

    def test_dirty_upstream_rejected(self):
        with patch('dllm_eval.baseline.subprocess.check_output',side_effect=[SOURCES['sparsed'][1],' M model.py']):
            with self.assertRaises(RuntimeError):source_manifest('sparsed','.')

    def test_reference_never_in_prompt(self):
        self.assertEqual(generation_prompt({'paper_prompt':'allowed','answer':'SECRET','solution':'SECRET','test':'SECRET'}),'allowed')

    def test_timing_difference_allowed(self):
        row=dict(token_ids=[1],text='x',raw_decoder_text='x',nfe=1,iterations=1,first_eos=None,output_tokens=1,seconds=1)
        assert_same_generation(row,dict(row,seconds=2))

    def test_token_difference_rejected(self):
        row=dict(token_ids=[1],text='x',raw_decoder_text='x',nfe=1,iterations=1,first_eos=None,output_tokens=1)
        with self.assertRaises(RuntimeError):assert_same_generation(row,dict(row,token_ids=[2]))


if __name__=='__main__':unittest.main()
