import json,tempfile,unittest
from pathlib import Path
import torch
from dllm_eval.dream_baselines import attention_context,settings
from dllm_eval.dream_campaign import METHODS,TOTAL,llada_ready,full_proof,accept,metrics,available_devices
from dllm_eval.plan import COUNTS


class DreamCampaignTests(unittest.TestCase):
    def test_fixed_requested_order(self):
        self.assertEqual(METHODS,('relay','d2cache','elastic_cache','fast_dllm_v1_no_flash'))
        self.assertEqual(TOTAL,13966)

    def test_flash_and_math_really_force_and_restore_backends(self):
        cuda=torch.backends.cuda
        flags=lambda:(cuda.flash_sdp_enabled(),cuda.math_sdp_enabled(),cuda.mem_efficient_sdp_enabled())
        initial=flags()
        for method,wanted in [('fast_dllm_v1_flash',(True,False,False)),('fast_dllm_v1_no_flash',(False,True,False))]:
            with self.assertRaisesRegex(ValueError,'cleanup'):
                with attention_context(method):
                    self.assertEqual(flags(),wanted);raise ValueError('cleanup')
            self.assertEqual(flags(),initial)
        a=settings('fast_dllm_v1_flash');b=settings('fast_dllm_v1_no_flash')
        self.assertNotEqual(a['attention_policy'],b['attention_policy'])
        for key in ('block_length','threshold','dual_cache','alg','seed','model','revision'):
            self.assertEqual(a[key],b[key])

    def proof(self):
        return dict(scored=TOTAL,cells={f'{t}_{n}':dict(samples=c,complete=True) for t,c in COUNTS.items() for n in (256,512)})

    def test_full_coverage_required(self):
        proof=self.proof();full_proof(proof)
        proof['cells']['math_512']['samples']-=1
        with self.assertRaises(RuntimeError):full_proof(proof)

    def write(self,path,value=None):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if value is not None else '0\n')

    def config(self,root):
        return dict(llada_output=str(root/'main'),llada_recovery=str(root/'recovery'),llada_v1_output=str(root/'v1'))

    def ready_main(self,root):
        for name in ('main/complete','main/exit_code','recovery/exit_code','recovery/v1_rearmed.json'):self.write(root/name)
        state=dict(status='complete',active={},completed=[dict(method=m,phase='full',rank=r,exit_code=0)
                        for m in ('d2cache','elastic_cache') for r in range(6)])
        self.write(root/'main/full_summary.json',state)
        for method in ('d2cache','elastic_cache'):
            self.write(root/'main'/f'{method}_full_summary.json',self.proof())
            for rank in range(6):self.write(root/'main/full'/method/f'rank{rank}/complete')

    def ready_v1(self,root):
        self.write(root/'v1/complete');self.write(root/'v1/exit_code')
        self.write(root/'v1/full_summary.json',dict(self.proof(),status='complete',active={},
            completed=[dict(phase='full',rank=r,exit_code=0) for r in range(6)]))
        for rank in range(6):self.write(root/f'v1/full/rank{rank}/complete')

    def test_waits_for_all_llada_including_v1(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);config=self.config(root)
            self.assertFalse(llada_ready(config));self.ready_main(root)
            self.assertFalse(llada_ready(config));self.ready_v1(root)
            self.assertTrue(llada_ready(config))
            (root/'v1/full/rank3/complete').unlink()
            with self.assertRaises(RuntimeError):llada_ready(config)

    def test_predecessor_failure_blocks_gpu_launch(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.ready_main(root);(root/'v1').mkdir();(root/'v1/exit_code').write_text('1\n')
            with self.assertRaisesRegex(RuntimeError,'failed'):llada_ready(self.config(root))

    def test_live_recovery_ignores_stale_original_failure_but_honors_new_failure(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'main').mkdir();(root/'main/exit_code').write_text('1\n')
            self.assertFalse(llada_ready(self.config(root)))
            (root/'recovery').mkdir();(root/'recovery/exit_code').write_text('1\n')
            with self.assertRaisesRegex(RuntimeError,'recovery failed'):llada_ready(self.config(root))

    def test_partial_v1_cannot_pass_with_marker_only(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.ready_main(root);self.ready_v1(root)
            proof=json.loads((root/'v1/full_summary.json').read_text());proof['scored']-=1
            self.write(root/'v1/full_summary.json',proof)
            with self.assertRaises(RuntimeError):llada_ready(self.config(root))

    def test_available_card_starts_without_waiting_for_all_and_cap_is_six(self):
        devices=[dict(index=g,uuid=f'GPU-{g}',used_mb=0,total_mb=32000,utilization=0) for g in range(8)]
        expected={str(g):f'GPU-{g}' for g in range(8)}
        self.assertEqual(len(available_devices(devices,[],{},expected)),6)
        for d in devices:d['used_mb']=1000
        devices[7]['used_mb']=0
        self.assertEqual([d['index'] for d in available_devices(devices,[],{},expected)],[7])
        self.assertFalse(available_devices(devices,[dict(uuid='GPU-7',pid=1)],{},expected))
        with self.assertRaises(RuntimeError):available_devices(devices,[],dict.fromkeys(range(7)),expected)

    def test_resume_replay_deduplicates_but_conflicting_rows_fail(self):
        row=dict(task='gsm8k',length=256,id='x',correct=True,seconds=2.,nfe=4)
        rows={};accept(rows,row,'relay');accept(rows,row,'relay')
        self.assertEqual(len(rows),1);self.assertEqual(metrics(rows)['gsm8k_256']['samples'],1)
        with self.assertRaises(RuntimeError):accept(rows,dict(row,seconds=3.),'relay')


if __name__=='__main__':unittest.main()
