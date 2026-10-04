import copy
import io
import json
import sys
import tempfile
import unittest
import warnings
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from dllm_eval import run
from dllm_eval.reporting import evaluation_log, result_table
from relay_cache.guards import SETTINGS, fingerprint


class ReportingTests(unittest.TestCase):
    def test_one_log_captures_stdout_stderr_progress_and_table(self):
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(out), redirect_stderr(err):
                with evaluation_log(Path(directory) / 'log', Path('runs/full')) as reporter:
                    print('model loading output')
                    print('stderr output', file=sys.stderr)
                    with warnings.catch_warnings():
                        warnings.simplefilter('always')
                        warnings.warn('visible warning')
                    reporter.start_cell('gsm8k', 256, 2)
                    reporter.update(dict(correct=True, seconds=.25, nfe=3))
                    reporter.update(dict(correct=False, seconds=.75, nfe=5))
                    reporter.finish_cell()
            logs = list((Path(directory) / 'log').glob('*.log'))
            self.assertEqual(len(logs), 1)
            text = logs[0].read_text(encoding='utf-8')
            for required in ('model loading output', 'stderr output', 'visible warning', '2/2',
                             '1/2', '50.00', '0.500', '4.00', 'Completed GSM8K'):
                self.assertIn(required, text)
            self.assertNotIn('\r', text)
            self.assertNotIn('\x1b', text)
            self.assertIn('2/2', err.getvalue())
            self.assertIn('Acc (%)', out.getvalue())

    def test_exception_and_interrupt_are_logged_and_streams_restored(self):
        for exception in (RuntimeError('test failure'), KeyboardInterrupt('test interrupt')):
            with self.subTest(exception=type(exception).__name__), tempfile.TemporaryDirectory() as directory:
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    with self.assertRaises(type(exception)):
                        with evaluation_log(directory, Path('run')) as reporter:
                            reporter.start_cell('math', 512, 2)
                            reporter.update(dict(correct=True, seconds=1., nfe=5))
                            raise exception
                    self.assertIs(sys.stdout, out)
                    self.assertIs(sys.stderr, err)
                text = next(Path(directory).glob('*.log')).read_text(encoding='utf-8')
                self.assertIn(type(exception).__name__, text)
                self.assertIn('--resume', text)
                self.assertNotIn('Completed MATH', text)
                self.assertNotIn('Evaluation complete.', text)

    def test_tables_are_read_only_and_zero_samples_are_not_perfect_accuracy(self):
        cells = {'humaneval_512': dict(samples=0, correct=0, accuracy_percent=None,
                                      mean_seconds=None, mean_nfe=None)}
        before = copy.deepcopy(cells)
        table = result_table(cells)
        self.assertEqual(cells, before)
        self.assertIn('0/0', table)
        self.assertIn('N/A', table)
        self.assertIn('HumanEval', table)
        self.assertNotIn('100.00', table)

    def test_terminal_redraws_but_log_has_only_plain_lines(self):
        class Terminal(io.StringIO):
            def isatty(self):return True
        out,err=Terminal(),Terminal()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(out), redirect_stderr(err):
            with evaluation_log(directory,Path('run')) as reporter:
                reporter.start_cell('mbpp',256,1)
                reporter.update(dict(correct=True,seconds=.25,nfe=10))
                reporter.finish_cell()
            text=next(Path(directory).glob('*.log')).read_text(encoding='utf-8')
            self.assertIn('\r',err.getvalue())
            self.assertNotIn('\r',text)
            self.assertNotIn('\x1b',text)
            self.assertIn('100.00',text)

    def test_resume_invocations_have_separate_single_transcripts(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for _ in range(2):
                with evaluation_log(directory, Path('run')):
                    print('invocation')
            logs = list(Path(directory).glob('*.log'))
            self.assertEqual(len(logs), 2)
            self.assertTrue(all(p.read_text(encoding='utf-8').count('invocation') == 1 for p in logs))

    def test_cli_logging_default_does_not_change_fixed_profile(self):
        args = run.parser().parse_args(['--gpu', '3', '--data-root', 'data', '--output', 'runs/full'])
        self.assertEqual(args.log_dir, run.ROOT / 'log')
        self.assertEqual(args.tasks, ['gsm8k', 'humaneval', 'mbpp', 'math'])
        self.assertEqual(args.lengths, [256, 512])
        self.assertIsNone(args.limit)


class RunnerPresentationRegressionTests(unittest.TestCase):
    def mocked_environment(self, stack, output, logdir, calls, fail_grade=None, tasks=None, lengths=None):
        samples = {task: [dict(id='a', paper_prompt='legal-a', answer='SECRET', solution='SECRET'),
                          dict(id='b', paper_prompt='legal-b', answer='SECRET', solution='SECRET')]
                   for task in (tasks or ['gsm8k', 'math'])}
        config = output.parent / 'config.json'
        config.write_text(json.dumps(dict(settings=SETTINGS)), encoding='utf-8')
        args = run.parser().parse_args(['--gpu', '3', '--data-root', 'unused', '--config', str(config),
                                       '--output', str(output), '--log-dir', str(logdir), '--tasks', *samples,
                                       '--lengths', *map(str, lengths or [256, 512])])

        class FakeSession:
            def __init__(self):
                calls.append(('load',))
                self.invocations = 0

            def prepare(self, prompt, task):
                calls.append(('prepare', prompt, task))
                return prompt

            def generate(self, prompt, length, sample, task):
                self.invocations += 1
                calls.append(('generate', task, length, sample['id']))
                return dict(seconds=100. if self.invocations % 2 else .5, nfe=3, iterations=3,
                            token_ids=[7, 126081], raw_decoder_text='answer', text='answer',
                            ordinary_calls=3, private_calls=0, ordinary_accepted=2,
                            actual_ordinary_row_layers=96, optional_row_layers_skipped=0,
                            phase_counts={'warm': 1}, boundary_bytes=64)

        def grade(text, sample, task):
            # The original runner must persist generation before accessing grading references.
            records = [json.loads(p.read_text(encoding='utf-8')) for p in output.glob(f'{task}_*/records/*.json')]
            row = [r for r in records if r['id'] == sample['id'] and 'assessment' not in r][-1]
            self.assertEqual(row['result']['text'], text)
            calls.append(('grade', task, row['length'], sample['id']))
            if fail_grade and sample['id'] == fail_grade:
                raise RuntimeError('scoring interrupted')
            return dict(correct=sample['id'] == 'a', policy='test only')

        stack.enter_context(patch('dllm_eval.run.verify_sources', return_value={'files': {}, 'artifacts': {}}))
        stack.enter_context(patch('dllm_eval.run.verify_environment', return_value={'test': 'mock'}))
        stack.enter_context(patch('dllm_eval.run.read_data', return_value=(samples, {'test': 'mock'})))
        stack.enter_context(patch('relay_cache.guards.exclusive_lock', side_effect=lambda path: nullcontext()))
        stack.enter_context(patch('relay_cache.guards.gpu_lease', side_effect=lambda gpu: nullcontext({'uuid': 'GPU-test'})))
        stack.enter_context(patch('relay_cache.guards.check_binding', return_value={'test': 'mock'}))
        stack.enter_context(patch('relay_cache.generate.Session', FakeSession))
        stack.enter_context(patch('dllm_eval.evaluation.evaluate', side_effect=grade))
        stack.enter_context(patch('dllm_eval.scoring_guard.installed', side_effect=lambda events: nullcontext()))
        stack.enter_context(patch('dllm_eval.run.parser', return_value=type('Parser', (), {'parse_args': lambda self: args})()))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        stack.enter_context(patch.dict('os.environ', {}))
        return args

    def test_fresh_and_resume_keep_warm_replay_order_grading_and_saved_bytes(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output, logs = Path(directory) / 'run', Path(directory) / 'log'
            calls = []
            args = self.mocked_environment(stack, output, logs, calls, tasks=['gsm8k','humaneval','mbpp','math'])
            run.main()
            expected = [('load',)]
            for length in (256, 512):
                for task in ('gsm8k','humaneval','mbpp','math'):
                    for ident in ('a', 'b'):
                        expected += [('prepare', 'legal-' + ident, task), ('generate', task, length, ident),
                                     ('generate', task, length, ident), ('grade', task, length, ident)]
            self.assertEqual(calls, expected)
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['scored'], 16)
            self.assertEqual(len(summary['cells']), 8)
            for cell in summary['cells'].values():
                self.assertEqual(cell, dict(samples=2, correct=1, accuracy_percent=50., mean_seconds=.5, mean_nfe=3.))
            before = {p.relative_to(output): p.read_bytes() for p in output.rglob('*.json')}
            calls.clear()
            args.resume = True
            run.main()
            self.assertEqual(calls, [])
            self.assertEqual(before, {p.relative_to(output): p.read_bytes() for p in output.rglob('*.json')})
            transcripts = list(logs.glob('*.log'))
            self.assertEqual(len(transcripts), 2)
            self.assertTrue(all('Final results:' in p.read_text(encoding='utf-8') for p in transcripts))

    def test_interrupted_grading_resumes_saved_generation_without_replaying_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output, logs = Path(directory) / 'run', Path(directory) / 'log'
            calls = []
            with ExitStack() as stack:
                self.mocked_environment(stack, output, logs, calls, fail_grade='b', tasks=['gsm8k'], lengths=[256])
                with self.assertRaisesRegex(RuntimeError, 'scoring interrupted'):
                    run.main()
            self.assertFalse((output / 'complete').exists())
            self.assertFalse((output / 'summary.json').exists())
            records = list(output.glob('gsm8k_256/records/*.json'))
            saved = {p: json.loads(p.read_text(encoding='utf-8'))['result'] for p in records}
            calls.clear()
            with ExitStack() as stack:
                args = self.mocked_environment(stack, output, logs, calls, tasks=['gsm8k'], lengths=[256])
                args.resume = True
                run.main()
            self.assertEqual(calls, [('grade', 'gsm8k', 256, 'b')])
            self.assertEqual(saved, {p: json.loads(p.read_text(encoding='utf-8'))['result'] for p in records})
            self.assertTrue((output / 'complete').exists())


if __name__ == '__main__':
    unittest.main()
