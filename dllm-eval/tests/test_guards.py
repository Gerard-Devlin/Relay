import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from relay_cache.utils import generation_prompt,select_samples
from relay_cache.guards import verify_sources,validate_resume
from relay_cache.utils import sha256
from relay_cache.guards import select_idle,check_binding,exclusive_lock
from relay_cache.execution import forbid_sdpa
from relay_cache.guards import SETTINGS,validate_settings
from dllm_eval.run import prepare_run

class GuardTests(unittest.TestCase):
    def test_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'relay_cache/a.py';p.parent.mkdir();p.write_text('a=1')
            m=dict(files={'relay_cache/a.py':sha256(p)},artifacts={})
            verify_sources(root,m);p.write_text('a=2')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):verify_sources(root,m)
    def test_live_source_changes_rejected_without_published_hash_catalog(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'relay_cache/a.py';p.parent.mkdir();p.write_text('a=1')
            baseline=verify_sources(root)
            baseline['files'].clear()
            self.assertIn('relay_cache/a.py',verify_sources(root)['files'])
            p.write_text('a=2')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):verify_sources(root)
    def test_live_config_file_changes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'dllm-eval/configs/profile.json';p.parent.mkdir(parents=True);p.write_text('{}')
            verify_sources(root)
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError,'artifact mismatch'):verify_sources(root)
    def test_live_source_file_set_changes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'relay_cache').mkdir();verify_sources(root)
            (root/'relay_cache/added.py').write_text('')
            with self.assertRaisesRegex(RuntimeError,'file set'):verify_sources(root)
    def test_source_file_set_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'relay_cache').mkdir();(root/'relay_cache/extra.py').write_text('')
            with self.assertRaisesRegex(RuntimeError,'file set'):verify_sources(root,dict(files={},artifacts={}))
    def test_prompt_gold_not_input(self):
        self.assertEqual(generation_prompt(dict(paper_prompt='legal',answer='SECRET',solution='SECRET',test='SECRET')),'legal')
        with self.assertRaises(ValueError):generation_prompt(dict(answer='SECRET'))
    def test_prompt_does_not_fallback_invalid_paper_field(self):
        with self.assertRaises(ValueError):generation_prompt(dict(paper_prompt='',prompt='fallback'))
    def test_selection_seed_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.json';p.write_text(json.dumps([dict(id=i,prompt='x') for i in range(10)]))
            self.assertEqual(select_samples(p,4,0),select_samples(p,4,0))
            p.write_text(json.dumps([dict(id='same')]*4))
            with self.assertRaises(ValueError):select_samples(p,4,0)
    def test_gpu_identity_and_occupation(self):
        r=dict(index=2,uuid='GPU-abc',used_mb=0,total_mb=32000,utilization=0)
        self.assertEqual(select_idle([r],'2')['uuid'],'GPU-abc')
        with self.assertRaises(RuntimeError):select_idle([r],'2',[dict(uuid='GPU-abc',pid=123)])
        with self.assertRaises(RuntimeError):select_idle([{**r,'used_mb':200}],'2')
    def test_binding_mismatch_before_cuda(self):
        with patch.dict(os.environ,{'RELAY_GPU_UUID':'GPU-abc','CUDA_VISIBLE_DEVICES':'GPU-other'}):
            with self.assertRaises(RuntimeError):check_binding()
    def test_duplicate_run_and_resume_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';prepare_run(p,{'hash':'same'})
            with self.assertRaises(FileExistsError):prepare_run(p,{'hash':'same'})
            prepare_run(p,{'hash':'same'},True)
            with self.assertRaises(RuntimeError):prepare_run(p,{'hash':'different'},True)
    def test_fixed_settings_reject_retune(self):
        validate_settings(SETTINGS)
        with self.assertRaises(ValueError):validate_settings({**SETTINGS,'threshold':.8})
    def test_sdpa_restored_on_exception(self):
        import torch
        old=torch.nn.functional.scaled_dot_product_attention
        with self.assertRaises(ValueError):
            with forbid_sdpa():raise ValueError('abort')
        self.assertIs(old,torch.nn.functional.scaled_dot_product_attention)
    @unittest.skipUnless(os.name=='posix','flock is Linux execution contract')
    def test_lock_conflict_and_exception_release(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'lock'
            with self.assertRaises(ValueError):
                with exclusive_lock(p):
                    with self.assertRaises(RuntimeError):
                        with exclusive_lock(p):pass
                    raise ValueError('abort')
            with exclusive_lock(p):pass
