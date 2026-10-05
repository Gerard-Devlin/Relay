import json,tempfile,unittest
from contextlib import ExitStack,nullcontext
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from dllm_eval import run,baseline_run
from dllm_eval.warmup import POLICY,requests,startup
import test_reporting


class StartupWarmupTests(unittest.TestCase):
    def test_only_two_startup_requests_and_only_legal_prompt_text(self):
        samples={'gsm8k':[dict(id=str(i),paper_prompt='legal',answer='SECRET') for i in range(3)],'math':[]}
        self.assertEqual([s['id'] for _,s in requests(samples,['gsm8k','math'])],['0','1'])
        one={'gsm8k':samples['gsm8k'][:1]}
        self.assertEqual(len(requests(one,['gsm8k'])),2)
        self.assertEqual(requests({'gsm8k':[]},['gsm8k']),[])
        with tempfile.TemporaryDirectory() as t:
            calls=[]
            session=SimpleNamespace(prepare=lambda text,task:calls.append(text) or text,
                generate=lambda *a:dict(seconds=99.,nfe=9))
            args=SimpleNamespace(tasks=['gsm8k'],lengths=[256,512],output=Path(t))
            startup(session,one,args,SimpleNamespace(info=lambda *a:None))
            self.assertEqual(calls,['legal','legal'])
            record=json.loads(next(Path(t).glob('warmup_*.json')).read_text())
            self.assertTrue(all(r['length']==256 for r in record['requests']))
            self.assertNotIn('SECRET',json.dumps(record))

    def test_partial_resume_warms_new_worker_but_never_regenerates_saved_answer(self):
        helper=test_reporting.RunnerPresentationRegressionTests()
        with tempfile.TemporaryDirectory() as t:
            out,logs=Path(t)/'run',Path(t)/'log';calls=[]
            with ExitStack() as stack:
                helper.mocked_environment(stack,out,logs,calls,fail_grade='a',tasks=['gsm8k'],lengths=[256])
                with self.assertRaisesRegex(RuntimeError,'scoring interrupted'):run.main()
            saved=next(out.glob('gsm8k_256/records/*.json'))
            answer=json.loads(saved.read_text())['result'];calls.clear()
            with ExitStack() as stack:
                args=helper.mocked_environment(stack,out,logs,calls,tasks=['gsm8k'],lengths=[256]);args.resume=True
                run.main()
            self.assertEqual([c for c in calls if c[0]=='generate'],
                [('generate','gsm8k',256,'a'),('generate','gsm8k',256,'b'),('generate','gsm8k',256,'b')])
            self.assertEqual(answer,json.loads(saved.read_text())['result'])
            self.assertEqual(json.loads((out/'summary.json').read_text())['scored'],2)
            self.assertEqual(len(list(out.glob('warmup_*.json'))),2)

    def test_failed_startup_cannot_produce_scored_records(self):
        helper=test_reporting.RunnerPresentationRegressionTests()
        with tempfile.TemporaryDirectory() as t,ExitStack() as stack:
            out,logs=Path(t)/'run',Path(t)/'log';calls=[]
            helper.mocked_environment(stack,out,logs,calls,tasks=['gsm8k'],lengths=[256])
            from relay_cache.llada.generate import Session
            original=Session.generate
            def fail(self,*args):
                if self.invocations==1:raise RuntimeError('warm-up failed')
                return original(self,*args)
            stack.enter_context(patch.object(Session,'generate',fail))
            with self.assertRaisesRegex(RuntimeError,'warm-up failed'):run.main()
            self.assertFalse(list(out.glob('*/records/*.json')))
            self.assertFalse((out/'complete').exists())
            self.assertFalse(any(c[0]=='grade' for c in calls))

    def test_baseline_counts_only_single_measured_generation_for_all_cells(self):
        helper=test_reporting.RunnerPresentationRegressionTests()
        with tempfile.TemporaryDirectory() as t,ExitStack() as stack:
            out,logs=Path(t)/'run',Path(t)/'log';calls=[]
            args=helper.mocked_environment(stack,out,logs,calls,tasks=['gsm8k','humaneval','mbpp','math'])
            args.method='d2cache';args.baseline_source=Path('unused')
            samples={task:[dict(id=i,paper_prompt='legal-'+i) for i in ('a','b')] for task in args.tasks}
            class Session:
                def __init__(self,*a):self.count=0
                def prepare_batch(self,texts,task):
                    if any('SECRET' in text for text in texts):raise AssertionError('Reference leaked')
                    return texts
                def generate_batch(self,prompt,length,group,task,batch_id):
                    self.count+=1;calls.append(('batch',task,length,group[0]['id']))
                    seconds=100. if self.count<=2 else .5
                    return [dict(text='answer',token_ids=[7],nfe=3,seconds=seconds,batch_seconds=seconds,
                        batch_size=1,batch_id=batch_id,method='d2cache')]
            for target,value in [('read_data',(samples,{})),('verify_sources',{}),('verify_environment',{}),
                                 ('source_manifest',{}),('verify_upstream',None)]:
                stack.enter_context(patch('dllm_eval.baseline_run.'+target,return_value=value))
            stack.enter_context(patch('dllm_eval.baseline_run.Session',Session))
            stack.enter_context(patch('dllm_eval.baseline_run.exclusive_lock',side_effect=lambda *a:nullcontext()))
            stack.enter_context(patch('dllm_eval.baseline_run.gpu_lease',side_effect=lambda *a:nullcontext({'uuid':'GPU-test'})))
            stack.enter_context(patch('dllm_eval.baseline_run.check_binding',return_value={}))
            stack.enter_context(patch('dllm_eval.baseline_run.installed',side_effect=lambda *a:nullcontext()))
            stack.enter_context(patch('dllm_eval.baseline_run.evaluate',side_effect=lambda text,sample,task:dict(correct=True)))
            from dllm_eval.baseline_batch import Events
            baseline_run.run_evaluation(args,Events())
            self.assertEqual(sum(c[0]=='batch' for c in calls),18)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['scored'],16);self.assertEqual(summary['measurement'],POLICY)
            self.assertTrue(all(c['mean_seconds']==.5 for c in summary['cells'].values()))
            self.assertTrue(all(c['mean_nfe']==3 for c in summary['cells'].values()))
            self.assertEqual(json.loads((out/'manifest.json').read_text())['measurement'],POLICY)
