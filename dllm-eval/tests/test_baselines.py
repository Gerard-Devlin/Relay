import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import ModuleType

from dllm_eval.baseline import settings, source_manifest, verify_upstream, assert_same_generation, SOURCES, official_imports, official_token_environment
from relay_cache.utils import sha256, generation_prompt


class BaselineTests(unittest.TestCase):
    def test_official_dllm_profile(self):
        value = settings('dllm_cache')
        self.assertEqual(value['per_task']['math'], [256,50,1])
        self.assertEqual(value['per_task']['gsm8k'], [8,50,7])
        self.assertEqual(value['per_task']['humaneval'], [32,50,8])
        self.assertEqual(value['per_task']['mbpp'], [32,100,5])
        self.assertEqual(value['transfer_ratio'], .25)
        self.assertFalse(value['parallel_decoding'])

    def test_official_d2cache_profile(self):
        value = settings('d2cache')
        self.assertEqual((value['rollout_p'],value['current_k'],value['sigma'],value['inflate_w']),(.1,32,10.,0))
        self.assertEqual(value['block'], 'gen_length')
        self.assertEqual(value['num_transfer_tokens'], 1)
        self.assertIn('eager', value['backend'])
        self.assertFalse(value['stop_until_eos'])
        self.assertFalse(value['parallel_decoding'])

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):settings('unknown')

    def test_official_cache_token_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            official_token_environment()
            self.assertEqual({k:os.environ[k] for k in ('MASK_TOKEN_ID','EOS_TOKEN_ID','PAD_TOKEN_ID')},
                             dict(MASK_TOKEN_ID='126336',EOS_TOKEN_ID='126081',PAD_TOKEN_ID='126081'))

    def test_upstream_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.py';p.write_text('original')
            manifest={'files':{'model.py':sha256(p)}}
            verify_upstream(tmp,manifest)
            p.write_text('changed')
            with self.assertRaises(RuntimeError):verify_upstream(tmp,manifest)

    def test_upstream_configuration_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.yaml';p.write_text('current_k: 32')
            manifest={'files':{'config.yaml':sha256(p)}}
            verify_upstream(tmp,manifest)
            p.write_text('current_k: 16')
            with self.assertRaises(RuntimeError):verify_upstream(tmp,manifest)

    def test_upstream_new_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'new.py').write_text('unexpected')
            with self.assertRaises(RuntimeError):verify_upstream(tmp,{'files':{}})

    def test_wrong_upstream_revision_rejected(self):
        with patch('dllm_eval.baseline.subprocess.check_output',return_value='wrong'):
            with self.assertRaises(RuntimeError):source_manifest('dllm_cache','.')

    def test_dirty_upstream_rejected(self):
        with patch('dllm_eval.baseline.subprocess.check_output',side_effect=[SOURCES['d2cache'][1],' M model.py']):
            with self.assertRaises(RuntimeError):source_manifest('d2cache','.')

    def test_reference_never_in_prompt(self):
        self.assertEqual(generation_prompt({'paper_prompt':'allowed','answer':'SECRET','solution':'SECRET','test':'SECRET'}),'allowed')

    def test_timing_difference_allowed(self):
        row=dict(token_ids=[1],text='x',raw_decoder_text='x',nfe=1,iterations=1,first_eos=None,output_tokens=1,seconds=1)
        assert_same_generation(row,dict(row,seconds=2))

    def test_token_difference_rejected(self):
        row=dict(token_ids=[1],text='x',raw_decoder_text='x',nfe=1,iterations=1,first_eos=None,output_tokens=1)
        with self.assertRaises(RuntimeError):assert_same_generation(row,dict(row,token_ids=[2]))

    def check_import_restore(self, fail):
        original_path=sys.path[:]
        old=ModuleType('antlr4');new=ModuleType('antlr4');child=ModuleType('antlr4.child')
        with patch.dict(sys.modules, {'antlr4':old}):
            try:
                with official_imports('/official/source','/private/deps'):
                    self.assertNotIn('antlr4',sys.modules)
                    sys.modules['antlr4']=new;sys.modules['antlr4.child']=child
                    if fail:raise ValueError('simulated import failure')
            except ValueError:
                self.assertTrue(fail)
            self.assertIs(sys.modules['antlr4'],old)
            self.assertNotIn('antlr4.child',sys.modules)
            self.assertEqual(sys.path,original_path)

    def test_scoring_runtime_restored_after_import(self):self.check_import_restore(False)
    def test_scoring_runtime_restored_after_import_failure(self):self.check_import_restore(True)


if __name__=='__main__':unittest.main()
