import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import ModuleType
from functools import wraps

from dllm_eval.baseline import settings, source_manifest, verify_upstream, assert_same_generation, SOURCES, official_imports, official_token_environment, elastic_transfers, left_pad, batches, batch_metrics, baseline_table
from relay_cache.utils import sha256, generation_prompt
from dllm_eval.baseline_batch import stage_jobs, current_method, restored_rows, MethodLogs, method_log_paths


class BaselineTests(unittest.TestCase):
    def test_official_elastic_profile(self):
        value=settings('elastic_cache')
        self.assertEqual((value['window_length'],value['threshold'],value['gamma'],value['track_num']), (16,.90,.90,1))
        self.assertTrue(value['block_caching']);self.assertTrue(value['stop_until_eos'])
        self.assertTrue(value['parallel_decoding']);self.assertEqual(value['batch_size'],1)
        self.assertNotIn('dllm_cache', SOURCES)

    def test_elastic_observer_preserves_return_and_restores_on_failure(self):
        class Positions:
            def numel(self):return 3
        output=(object(),Positions(),object())
        namespace={'get_decoded_token_confident':lambda:output}
        class Generator:__globals__=namespace
        original=namespace['get_decoded_token_confident']
        with self.assertRaises(ValueError):
            with elastic_transfers(Generator) as counts:
                self.assertIs(namespace['get_decoded_token_confident'](),output)
                self.assertEqual(counts,[3]);raise ValueError('probe')
        self.assertIs(namespace['get_decoded_token_confident'],original)

    def test_official_d2cache_profile(self):
        value = settings('d2cache')
        self.assertEqual((value['rollout_p'],value['current_k'],value['sigma'],value['inflate_w']),(.1,32,10.,4))
        self.assertEqual((value['block'],value['generation_sigma'],value['threshold']), (32,0.,.90))
        self.assertEqual(value['num_transfer_tokens'], 1)
        self.assertIn('eager', value['backend'])
        self.assertFalse(value['stop_until_eos'])
        self.assertTrue(value['parallel_decoding'])

    def test_elastic_decorated_generator_observer(self):
        class Positions:
            def numel(self): return 2
        output=(object(), Positions(), object())
        namespace={'get_decoded_token_confident':lambda:output}
        exec('def generate(): return get_decoded_token_confident()', namespace)
        original=namespace['get_decoded_token_confident']
        @wraps(namespace['generate'])
        def decorated(): return namespace['generate']()
        self.assertNotIn('get_decoded_token_confident', decorated.__globals__)
        with elastic_transfers(decorated) as counts:
            self.assertIs(decorated(), output)
            self.assertEqual(counts, [2])
        self.assertIs(namespace['get_decoded_token_confident'], original)

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):settings('unknown')

    def test_algorithm_logs_isolate_workers_and_preserve_progress_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=method_log_paths(tmp,'runs/campaign')
            with MethodLogs(paths) as logs:
                logs.write('Shared model and scoring profile\n')
                for gpu in (1,3,4,5,6,7):logs.write(f'd2 worker GPU{gpu}\n','d2cache')
                logs.write('d2 100%|\u2588\u2588| 2/2\n| Acc (%) | 50.00 |\n','d2cache')
                logs.write('Elastic worker GPU0\n','elastic_cache')
            d2=paths['d2cache'].read_text(encoding='utf-8');elastic=paths['elastic_cache'].read_text(encoding='utf-8')
            self.assertIn('GPU7',d2);self.assertIn('\u2588',d2);self.assertIn('Acc (%)',d2)
            self.assertNotIn('Elastic worker',d2);self.assertNotIn('d2 worker',elastic)
            self.assertIn('Shared model and scoring profile',elastic)
            self.assertEqual(len(list(Path(tmp).glob('*.log'))),2)
            with self.assertRaises(FileExistsError):
                with MethodLogs(paths):pass
            with MethodLogs(paths,resume=True) as logs:logs.write('resume\n','elastic_cache')
            self.assertNotIn('resume',paths['d2cache'].read_text(encoding='utf-8'))
            self.assertIn('resume',paths['elastic_cache'].read_text(encoding='utf-8'))

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
            with self.assertRaises(RuntimeError):source_manifest('elastic_cache','.')

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

    def test_full_schedule_runs_six_d2cache_shards_first(self):
        jobs=stage_jobs('full',6)
        self.assertEqual(jobs[:6],[dict(method='d2cache',rank=r) for r in range(6)])
        self.assertEqual(jobs[6:],[dict(method='elastic_cache',rank=r) for r in range(6)])

    def test_method_barrier_waits_for_active_d2cache(self):
        jobs=[dict(method='elastic_cache',rank=r) for r in range(6)]
        running={1:(dict(method='d2cache',rank=5),object())}
        self.assertEqual(current_method(jobs,running),'d2cache')
        self.assertEqual(current_method(jobs,{}),'elastic_cache')
        self.assertIsNone(current_method([],{}))

    def make_record(self, root, assessment=True):
        rank=Path(root)/'full/d2cache/rank0'
        path=rank/'gsm8k_256/records/a.json'
        path.parent.mkdir(parents=True)
        manifest=rank/'manifest.json';manifest.write_text('{"source":"frozen"}')
        row=dict(task='gsm8k',length=256,id='sample',manifest_sha256=sha256(manifest),
                 result=dict(method='d2cache',seconds=12.,nfe=30,batch_id='batch',batch_seconds=12.,batch_size=1))
        if assessment:row['assessment']=dict(correct=True)
        path.write_text(json.dumps(row))
        return path,manifest

    def test_resume_restores_assessed_records_without_modifying_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=self.make_record(tmp);before=path.read_bytes()
            rows=restored_rows(Path(tmp),'full')
            self.assertEqual(rows[('d2cache','gsm8k',256,'sample')],dict(task='gsm8k',length=256,id='sample',correct=True,seconds=12.,nfe=30,method='d2cache',batch_id='batch',batch_seconds=12.,batch_size=1))
            self.assertEqual(path.read_bytes(),before)

    def test_resume_preserves_unscored_generation_for_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=self.make_record(tmp,assessment=False);before=path.read_bytes()
            self.assertEqual(restored_rows(Path(tmp),'full'),{})
            self.assertEqual(path.read_bytes(),before)

    def test_resume_rejects_changed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,manifest=self.make_record(tmp);manifest.write_text('changed')
            with self.assertRaises(RuntimeError):restored_rows(Path(tmp),'full')

    def test_resume_rejects_duplicate_prompt_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=self.make_record(tmp)
            (path.parent/'duplicate.json').write_bytes(path.read_bytes())
            with self.assertRaises(RuntimeError):restored_rows(Path(tmp),'full')

    def test_official_left_padding_preserves_each_prompt(self):
        ids,mask=left_pad([[10,11],[20]],99)
        self.assertEqual(ids,[[10,11],[99,20]])
        self.assertEqual(mask,[[1,1],[0,1]])
        for tokens,attn,original in zip(ids,mask,[[10,11],[20]]):
            self.assertEqual([x for x,m in zip(tokens,attn) if m],original)

    def test_true_batch_eight_and_final_partial_batch(self):
        self.assertEqual(list(map(len,batches(list(range(19)),8))),[8,8,3])
        with self.assertRaises(ValueError):batches([1],0)
        with self.assertRaises(ValueError):left_pad([[]],0)

    def test_batch_latency_not_reported_as_single_request_latency(self):
        rows=[dict(batch_id='eight',batch_size=8,batch_seconds=16.,seconds=2.,nfe=256,correct=True) for _ in range(8)]
        rows.append(dict(batch_id='partial',batch_size=1,batch_seconds=3.,seconds=3.,nfe=256,correct=False))
        result=batch_metrics(rows)
        self.assertEqual(result['samples'],9)
        self.assertEqual(result['batches'],2)
        self.assertAlmostEqual(result['mean_seconds'],19/9)
        self.assertEqual(result['mean_batch_seconds'],9.5)
        self.assertAlmostEqual(result['throughput_requests_per_second'],9/19)
        self.assertEqual(result['actual_batch_sizes'],[1,8])
        table=baseline_table({'gsm8k_256':result})
        self.assertIn('Avg batch (s)',table)
        self.assertNotIn('Request (s)',table)

    def test_inconsistent_or_wrong_amortized_timing_is_rejected(self):
        row=dict(batch_id='x',batch_size=8,batch_seconds=16.,seconds=2.,nfe=256,correct=True)
        with self.assertRaises(RuntimeError):batch_metrics([row,dict(row,batch_seconds=24.,seconds=3.)])
        with self.assertRaises(RuntimeError):batch_metrics([dict(row,seconds=16.)])


if __name__=='__main__':unittest.main()
