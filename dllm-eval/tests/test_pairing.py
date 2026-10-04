"""Reject incomplete, cross-GPU or duplicated pairing before interpreting scores."""
import copy
import unittest
from dllm_eval.pairing import TASKS,METHODS,finish_report

def fixture():
    return dict(rows=[dict(task=t,id=str(i),method=m,gpu=2+i%6,seconds=1.0,nfe=1,
        correct=True,ordinary_calls=1,private_calls=0,private_accepted=0,output_tokens=1,
        truncated=False,actual_ordinary_row_layers=32,optional_row_layers_skipped=0,boundary_bytes=0)
        for t in TASKS for i in range(32) for m in METHODS])

class PairingTests(unittest.TestCase):
    def test_cross_gpu_pair_rejected(self):
        report=fixture();report['rows'][2]['gpu']=7
        with self.assertRaises(AssertionError):finish_report(report)
    def test_duplicate_method_rejected(self):
        report=fixture();report['rows'][2]['method']='flash_cache'
        with self.assertRaises(AssertionError):finish_report(report)
    def test_quality_regression_fails_fixed_gate(self):
        report=fixture()
        for row in report['rows']:
            if row['method']=='relay_cache':
                row['seconds']=.8
                if row['task']=='math' and int(row['id'])<3:row['correct']=False
        result=finish_report(report)
        self.assertTrue(result['gates']['relay_cache']['parts']['speed'])
        self.assertFalse(result['gates']['relay_cache']['parts']['point_quality'])
        self.assertFalse(result['gates']['relay_cache']['parts']['task_quality'])
        self.assertFalse(result['continuing_gate'])
        self.assertFalse(result['goal_achieved'])

if __name__=='__main__':unittest.main()
