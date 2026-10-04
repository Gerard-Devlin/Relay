import unittest
from relay_cache.cache import Plan,block_table,validate_identity,theoretical_row_layers

class TestRelay(unittest.TestCase):
    def test_active_rows_never_skip(self):
        p=Plan(tuple(range(8)),(3,5,7),3,2)
        for layer in range(32):self.assertTrue(set(p.live_rows)<=set(p.compute_rows(layer)))
        self.assertEqual(sum(len(p.compute_rows(l)) for l in range(32)),5*32+3*8)
    def test_each_stripe_refreshes_once_in_four_calls(self):
        plans=[Plan(tuple(range(8)),(3,5,7),3,p) for p in range(4)]
        for layer in range(32):self.assertEqual(sum(7 in p.compute_rows(layer) for p in plans),1)
    def test_changed_identity_cannot_relay(self):
        with self.assertRaises(AssertionError):validate_identity((2,4),(1,),(10,20),{4:21},())
        with self.assertRaises(AssertionError):validate_identity((2,4),(1,),(10,20),{4:20},(4,))
    def test_unchanged_identity_is_only_a_provenance_check(self):
        validate_identity((2,4),(1,),(10,20),{4:20},())
    def test_ragged_tables_preserve_mask_then_history(self):
        m,t=block_table(17,53,512)
        self.assertEqual(m,((0,512,0,17),));self.assertEqual(t,((0,512,17,49),(0,512,49,53)))
    def test_full_control_never_skips(self):
        p=Plan(tuple(range(8)),(3,5,7),3,1)
        self.assertTrue(all(p.compute_rows(l,True)==tuple(range(8)) for l in range(32)))
    def test_planner_rejects_mask_as_optional(self):
        with self.assertRaises(AssertionError):Plan(tuple(range(8)),(1,7),3,1)
    def test_theory_is_not_latency(self):
        cost=theoretical_row_layers(128,128)
        self.assertEqual(cost['full'],8192);self.assertEqual(cost['relay'],5120)

if __name__=='__main__':unittest.main()
