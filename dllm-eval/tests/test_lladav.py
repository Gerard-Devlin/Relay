"""Visual-input isolation, immutable profiles, recovery and scorer regressions."""
import copy,json,tempfile,types,unittest,io
from contextlib import ExitStack,nullcontext,redirect_stdout
from pathlib import Path
from unittest.mock import patch
from relay_cache.lladav.settings import SETTINGS,validate_settings
from relay_cache.lladav.priming import native_visual_priming
from dllm_eval.visual_data import official_query
from dllm_eval.visual_output import for_scoring
from dllm_eval.visual_run import image_path,validate_record
from dllm_eval.visual_score import grade,binary,extraction_prompt

class VisualTests(unittest.TestCase):
    def vista(self):
        return dict(question='How many sides?',question_type='multi_choice',answer_type='text',precision=0,
            choices=['3','4','5'],unit='',answer='4',evaluation_query='How many sides?')
    def test_reference_and_solution_are_not_prompt_inputs(self):
        for task,doc in [('mathvista',self.vista()),('mathverse',dict(question='Find x.',question_type='free-form',query_wo='Find x.'))]:
            original=official_query(task,doc)
            changed=dict(doc,answer='SECRET_REFERENCE',solution='SECRET_SOLUTION',query_cot='SECRET_REASONING')
            self.assertEqual(original,official_query(task,changed))
            self.assertNotIn('SECRET',original)
    def test_official_choices_and_reason_first_hint(self):
        prompt=official_query('mathvista',self.vista())
        self.assertIn('(B) 4',prompt);self.assertIn('Answer: xxx',prompt)
    def test_download_uses_pinned_revisions_and_only_visual_splits(self):
        from dllm_eval.visual_data import download_snapshots
        with patch('huggingface_hub.snapshot_download') as download:
            download_snapshots('/cache')
        calls=download.call_args_list
        self.assertEqual(len(calls),4)
        self.assertEqual(calls[0].kwargs['revision'],SETTINGS['revision'])
        self.assertEqual(calls[1].kwargs['revision'],SETTINGS['vision_revision'])
        self.assertTrue(all(c.kwargs.get('repo_type')=='dataset' for c in calls[2:]))
        self.assertTrue(all('testmini' in c.kwargs['allow_patterns'][0] for c in calls[2:]))
    def test_mathverse_uses_query_wo(self):
        self.assertEqual(official_query('mathverse',dict(question='Find x.',question_type='free-form',query_wo='Find x. Answer directly.')),'Find x. Answer directly.')
    def test_changed_algorithm_profile_is_rejected(self):
        for key,value in [('history_budget',64),('threshold',.8),('stripe_width',16),('verify',True),('revision','changed')]:
            bad=copy.deepcopy(SETTINGS);bad[key]=value
            with self.assertRaises(ValueError):validate_settings(bad)
    def test_image_escape_is_rejected(self):
        root=Path(tempfile.gettempdir())/'visual_data'
        self.assertEqual(image_path(root,'images/a.png'),(root/'images/a.png').resolve())
        for path in ('../private.png',str(Path(tempfile.gettempdir()).resolve()/'private.png')):
            with self.assertRaises(ValueError):image_path(root,path)
    def test_resume_rejects_wrong_identity_prompt_or_pixels(self):
        from relay_cache.guards import fingerprint
        sample=dict(id='1',prompt='question',image_sha256='pixels')
        good=dict(identity='run',id='1',prompt_sha256=fingerprint('question'),image_sha256='pixels',result=dict(token_ids=[10],nfe=1))
        validate_record(good,sample,'run')
        for key in ('identity','id','prompt_sha256','image_sha256'):
            bad=dict(good);bad[key]='changed'
            with self.assertRaises(RuntimeError):validate_record(bad,sample,'run')
    def test_native_priming_restores_after_exception(self):
        engine=types.SimpleNamespace(original=lambda *args:'ordinary',call={'first':False})
        native=types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace())]))
        old=engine.original
        with self.assertRaisesRegex(RuntimeError,'generation failed'):
            with native_visual_priming(engine,native):
                self.assertEqual(engine.original(None,None,1,None,None),'ordinary')
                raise RuntimeError('generation failed')
        self.assertIs(engine.original,old)
    def test_stop_boundary_retains_raw_tokens(self):
        tokenizer=types.SimpleNamespace(encode=lambda *a,**k:[8,9],decode=lambda ids,**k:','.join(map(str,ids)))
        result=dict(token_ids=[2,3,8,9,4]);text,completion=for_scoring(result,tokenizer,'stop',256)
        self.assertEqual(text,'2,3');self.assertEqual(result['token_ids'],[2,3,8,9,4])
        self.assertEqual(completion['reason'],'conversation_stop')
    def test_eos_precedes_conversation_stop(self):
        tokenizer=types.SimpleNamespace(encode=lambda *a,**k:[8],decode=lambda ids,**k:str(ids))
        _,completion=for_scoring(dict(token_ids=[1,126081,8]),tokenizer,'stop',256)
        self.assertEqual(completion['scored_tokens'],1);self.assertEqual(completion['reason'],'eos')
    def test_extraction_excludes_reference(self):
        calls=[]
        def client(body,stage):calls.append(body);return 'B'
        doc=self.vista();doc['answer']='SECRET_REFERENCE'
        grade('mathvista','The answer is B.',doc,client)
        self.assertNotIn('SECRET_REFERENCE',json.dumps(calls))
    def test_mathvista_normalization_matches_original_policy(self):
        self.assertTrue(grade('mathvista','The answer is B.',self.vista(),lambda *a:'B')['correct'])
        self.assertFalse(grade('mathvista','The answer is A.',self.vista(),lambda *a:'A')['correct'])
    def test_api_failure_remains_pending(self):
        def failed(*args):raise RuntimeError('API failed')
        with self.assertRaisesRegex(RuntimeError,'API failed'):grade('mathvista','The answer is B.',self.vista(),failed)
    def test_mathverse_literal_space_tail_and_binary_rule(self):
        self.assertIn('a\nb',extraction_prompt('a\nb'))
        self.assertTrue(binary('Judgement: 1'))
        for bad in ('maybe','0 and 1',''):
            with self.assertRaises(ValueError):binary(bad)
    def test_visual_resume_preserves_generation_and_duplicate_is_rejected(self):
        from PIL import Image
        from dllm_eval import visual_run
        from relay_cache.guards import ROOT
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);image=root/'image.png';Image.new('RGB',(2,2)).save(image)
            (root/'data.json').write_text('{}')
            sample=dict(id='1',task='mathvista',prompt='question',image='image.png',image_sha256='pixels')
            calls=[]
            def generate(value,length):
                calls.append(length)
                return dict(token_ids=[2,8,9],seconds=.5,nfe=2)
            tokenizer=types.SimpleNamespace(encode=lambda *a,**k:[8,9],decode=lambda ids,**k:str(ids))
            session=types.SimpleNamespace(loading={},mapping={},tokenizer=tokenizer,
                prepare=lambda *a:dict(image_sha256='pixels',stop='stop'),generate=generate)
            stack.enter_context(patch.object(visual_run,'verify_sources',return_value={}))
            stack.enter_context(patch.object(visual_run,'verify_environment',return_value={}))
            stack.enter_context(patch.object(visual_run,'read_data',return_value={'mathvista':[sample]}))
            stack.enter_context(patch('relay_cache.lladav.environment.verify_visual_environment',return_value={}))
            stack.enter_context(patch('relay_cache.guards.exclusive_lock',side_effect=lambda *a:nullcontext()))
            stack.enter_context(patch('relay_cache.guards.gpu_lease',side_effect=lambda *a:nullcontext({})))
            stack.enter_context(patch('relay_cache.guards.check_binding',return_value={}))
            stack.enter_context(patch('relay_cache.lladav.generate.Session',return_value=session))
            for name in ('synchronize','reset_peak_memory_stats'):
                stack.enter_context(patch('torch.cuda.'+name))
            for name in ('max_memory_allocated','max_memory_reserved'):
                stack.enter_context(patch('torch.cuda.'+name,return_value=1))
            args=types.SimpleNamespace(config=ROOT/'dllm-eval/configs/lladav.json',data_root=root,
                hf_hub_cache=root,tasks=['mathvista'],lengths=[256],limit=1,rank=0,world_size=1,
                output=root/'run',resume=False,gpu='2')
            reporter=types.SimpleNamespace(progress_stream=io.StringIO(),info=lambda *a:None)
            stack.enter_context(redirect_stdout(io.StringIO()))
            visual_run.run(args,reporter)
            self.assertEqual(calls,[256,256,256])
            path=next((args.output/'records').rglob('*.json'));before=path.read_bytes()
            with self.assertRaises(FileExistsError):visual_run.run(args,reporter)
            calls.clear();args.resume=True;visual_run.run(args,reporter)
            self.assertEqual(calls,[256,256]);self.assertEqual(path.read_bytes(),before)

if __name__=='__main__':unittest.main()
