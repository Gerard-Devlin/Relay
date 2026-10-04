"""Narrow compatibility repair for malformed Union(Boolean) parser objects.

Keep extraction and ordinary Math-Verify calls unchanged. Only the documented
AttributeError gets a real-set canonicalization with an exact partition check.
An unsupported expression or another error remains an explicit failure.
"""
from contextlib import contextmanager
from functools import cmp_to_key


def canonical_real_set(value):
    from sympy import Union
    from sympy.core.relational import Relational
    from sympy.logic.boolalg import BooleanFunction
    if isinstance(value, Union):
        return Union(*(canonical_real_set(child) for child in value.args))
    if isinstance(value, (BooleanFunction, Relational)):
        if len(value.free_symbols) > 1:
            raise ValueError('Multiple variables have no certified real-set interpretation')
        return value.as_set()
    return value


def exact_real_set_equal(left, right):
    from sympy import Union, Interval, FiniteSet, S, oo, simplify, Equivalent
    def endpoints(value):
        if value in (S.EmptySet, S.Reals):
            return []
        if isinstance(value, Union):
            return [point for child in value.args for point in endpoints(child)]
        if isinstance(value, Interval):
            return [point for point in (value.start, value.end) if point not in (-oo, oo)]
        if isinstance(value, FiniteSet):
            if not all(point.is_real is True and point.is_number is True for point in value):
                raise ValueError('Finite set is not composed of certified real numbers')
            return list(value)
        raise TypeError('Unsupported real-set object')
    def compare(a, b):
        delta = simplify(a-b)
        if delta.is_zero is True:
            return 0
        if delta.is_positive is True:
            return 1
        if delta.is_negative is True:
            return -1
        raise ValueError('Endpoint order not certified')
    bounds = set(endpoints(left)+endpoints(right))
    if not all(point.is_real is True and point.is_number is True for point in bounds):
        raise ValueError('Endpoint is not a certified real number')
    ordered = sorted(bounds, key=cmp_to_key(compare))
    probes = list(ordered)+[(a+b)/2 for a,b in zip(ordered, ordered[1:])]
    probes += [ordered[0]-1, ordered[-1]+1] if ordered else [S.Zero]
    decisions = [simplify(Equivalent(left.contains(point), right.contains(point))) for point in probes]
    if not all(value in (S.true, S.false) for value in decisions):
        raise ValueError('Membership comparison not certified')
    return all(value is S.true for value in decisions)


@contextmanager
def installed(events):
    from dllm_eval.score_answers import MathComparison
    original = MathComparison.equivalent
    def equivalent(self, left, right):
        try:
            return original(self, left, right)
        except AttributeError as error:
            if "'And' object has no attribute 'is_subset'" not in str(error):
                raise
            a,b = canonical_real_set(left),canonical_real_set(right)
            certificate = exact_real_set_equal(a,b)
            # The original comparison program must independently agree after
            # replacing the malformed parser objects by their exact real sets.
            checked = original(self,a,b)
            if certificate != checked:
                raise AssertionError('Exact set certificate and Math-Verify disagree')
            events.append(dict(error='Union(Boolean) subset AttributeError',
                equivalent=certificate,exact_partition_certificate=True,
                extraction_changed=False,numeric_approximation=False))
            return certificate
    MathComparison.equivalent = equivalent
    try:
        yield
    finally:
        MathComparison.equivalent = original
