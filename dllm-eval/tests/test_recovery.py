import unittest
import sys,types
from unittest.mock import patch
from sympy import Symbol,And,Or,Union,Interval,FiniteSet,S,sqrt
from dllm_eval.scoring_guard import canonical_real_set,exact_real_set_equal,installed
from dllm_eval.recovery import split_missing,PHYSICAL
from dllm_eval.plan import COUNTS,CELLS,sample_id


class SetRepairTests(unittest.TestCase):
    def test_interval_union_boolean_representation(self):
        x=Symbol('x',real=True)
        malformed=Union(And(x>0,x<1),Union(And(x>2,x<3),And(x>4,x<5),evaluate=False),evaluate=False)
        reference=Union(Interval.open(0,1),Interval.open(2,3),Interval.open(4,5))
        self.assertTrue(exact_real_set_equal(reference,canonical_real_set(malformed)))
    def test_open_closed_endpoint_difference(self):
        self.assertFalse(exact_real_set_equal(Interval.open(0,1),Interval(0,1)))
    def test_missing_interior_component(self):
        self.assertFalse(exact_real_set_equal(Union(Interval(0,1),Interval(2,3)),Interval(0,3)))
    def test_singleton_and_interval_boundary(self):
        self.assertTrue(exact_real_set_equal(Union(Interval.open(0,1),FiniteSet(0,1)),Interval(0,1)))
    def test_unbounded_empty_and_real(self):
        self.assertTrue(exact_real_set_equal(S.EmptySet,S.EmptySet))
        self.assertFalse(exact_real_set_equal(S.EmptySet,S.Reals))
        self.assertFalse(exact_real_set_equal(Interval(-S.Infinity,0),Interval(-S.Infinity,1)))
    def test_exact_algebraic_endpoint(self):
        self.assertFalse(exact_real_set_equal(Interval(0,sqrt(2)),Interval(0,sqrt(3))))
    def test_multivariate_not_certified(self):
        x,y=Symbol('x',real=True),Symbol('y',real=True)
        with self.assertRaises(ValueError):canonical_real_set(And(x>0,y<1))
    def test_unknown_finite_elements_not_certified(self):
        with self.assertRaises(ValueError):exact_real_set_equal(FiniteSet(Symbol('x')),FiniteSet(1))

    def test_repair_hook_normal_and_known_error(self):
        class Comparison:
            def equivalent(self,a,b):
                if isinstance(b,Union) and any(isinstance(child,And) for child in b.args):
                    raise AttributeError("'And' object has no attribute 'is_subset'")
                return exact_real_set_equal(a,b)
        module=types.ModuleType('dllm_eval.score_answers');module.MathComparison=Comparison
        original=Comparison.equivalent;events=[];x=Symbol('x',real=True)
        malformed=Union(And(x>0,x<1),And(x>2,x<3),evaluate=False)
        reference=Union(Interval.open(0,1),Interval.open(2,3))
        with patch.dict(sys.modules,{'dllm_eval.score_answers':module}):
            with installed(events):
                self.assertTrue(Comparison().equivalent(Interval(0,1),Interval(0,1)))
                self.assertEqual(events,[])
                self.assertTrue(Comparison().equivalent(reference,malformed))
        self.assertIs(Comparison.equivalent,original);self.assertEqual(len(events),1)

    def test_unrelated_error_raises_and_restores(self):
        class Comparison:
            def equivalent(self,a,b):raise AttributeError('Unrelated error')
        module=types.ModuleType('dllm_eval.score_answers');module.MathComparison=Comparison
        original=Comparison.equivalent;events=[]
        with patch.dict(sys.modules,{'dllm_eval.score_answers':module}):
            with self.assertRaisesRegex(AttributeError,'Unrelated error'):
                with installed(events):Comparison().equivalent(1,2)
        self.assertIs(Comparison.equivalent,original);self.assertEqual(events,[])

    def test_disagreeing_certificate_cannot_be_committed(self):
        class Comparison:
            def equivalent(self,a,b):
                if isinstance(b,Union) and any(isinstance(child,And) for child in b.args):
                    raise AttributeError("'And' object has no attribute 'is_subset'")
                return True
        module=types.ModuleType('dllm_eval.score_answers');module.MathComparison=Comparison
        events=[];x=Symbol('x',real=True);malformed=Union(And(x>0,x<1),And(x>2,x<3),evaluate=False)
        with patch.dict(sys.modules,{'dllm_eval.score_answers':module}):
            with self.assertRaisesRegex(AssertionError,'disagree'):
                with installed(events):Comparison().equivalent(Interval(0,3),malformed)
        self.assertEqual(events,[])


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets={task:[dict(id=f'{task}:{i}') for i in range(n)] for task,n in COUNTS.items()}
    def test_full_missing_balanced_exact_coverage(self):
        assignments=split_missing(self.datasets,[])
        self.assertEqual(sum(len(rows) for workers in assignments.values() for tasks in workers.values() for rows in tasks.values()),13966)
        for workers in assignments.values():
            for task in COUNTS:
                sizes=[len(workers[str(g)][task]) for g in PHYSICAL]
                self.assertLessEqual(max(sizes)-min(sizes),1)
    def test_partial_saved_generation_not_regenerated(self):
        saved=[(task,length,sample_id(rows[0])) for task,length in CELLS for rows in [self.datasets[task]]]
        assignments=split_missing(self.datasets,saved)
        actual={(task,int(length),row['id']) for length,workers in assignments.items() for tasks in workers.values() for task,rows in tasks.items() for row in rows}
        self.assertTrue(set(saved).isdisjoint(actual))
        self.assertEqual(len(actual),13966-len(saved))
    def test_complete_needs_no_generation(self):
        saved=[(task,length,sample_id(row)) for task,length in CELLS for row in self.datasets[task]]
        assignments=split_missing(self.datasets,saved)
        self.assertFalse(any(rows for workers in assignments.values() for tasks in workers.values() for rows in tasks.values()))
    def test_duplicate_and_unknown_rejected(self):
        with self.assertRaises(ValueError):split_missing(self.datasets,[('math',256,'math:0')]*2)
        with self.assertRaises(ValueError):split_missing(self.datasets,[('math',256,'unknown')])


if __name__=='__main__':unittest.main()
