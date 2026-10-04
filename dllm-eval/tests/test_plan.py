import unittest
from dllm_eval.plan import COUNTS,CELLS,SETTINGS,GPUS,shard,cell_summary

class CoverageTests(unittest.TestCase):
    def test_exact_coverage_and_balance_for_full_datasets(self):
        for count in COUNTS.values():
            rows=[{'id':str(i)} for i in range(count)]
            parts=[shard(rows,g) for g in GPUS]
            assigned=[i for part in parts for i,_ in part]
            self.assertEqual(len(assigned),count)
            self.assertEqual(set(assigned),set(range(count)))
            self.assertLessEqual(max(map(len,parts))-min(map(len,parts)),1)
            self.assertEqual(parts,[shard(rows,g) for g in GPUS])
    def test_duplicates_and_unauthorized_gpu_rejected(self):
        with self.assertRaises(ValueError):shard([{'id':'x'},{'id':'x'}],2)
        with self.assertRaises(ValueError):shard([{'id':'x'}],0)
    def test_all_eight_cells_and_count(self):
        self.assertEqual(len(CELLS),8)
        self.assertEqual(sum(COUNTS[t] for t,_ in CELLS),13966)
        self.assertEqual(SETTINGS['verify'],False)
        self.assertEqual(SETTINGS['stripe_width'],8)
    def test_duplicate_score_cannot_finish_cell(self):
        row=dict(id='x',correct=True,seconds=1,nfe=1,ordinary_calls=1,private_calls=0,
            truncated=False,output_tokens=1,actual_ordinary_row_layers=32,
            optional_row_layers_skipped=0,boundary_bytes=0)
        with self.assertRaises(ValueError):cell_summary([row,row],2)

if __name__=='__main__':unittest.main()
