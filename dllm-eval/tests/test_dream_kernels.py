import unittest
import torch
from relay_cache.dream.kernels import prepare


def repeat(x,groups):
    batch,heads,rows,dim=x.shape
    return x[:,:,None,:,:].expand(batch,heads,groups,rows,dim).reshape(batch,heads*groups,rows,dim)


def rotate(q,k,cos,sin):
    def apply(x):
        left,right=x.chunk(2,-1)
        return x*cos[:,None]+torch.cat((-right,left),-1)*sin[:,None]
    return apply(q),apply(k)


class GroupedCacheTests(unittest.TestCase):
    def test_full_and_partial_updates_are_exact_and_keep_other_positions(self):
        torch.manual_seed(71)
        cos,sin=torch.randn(1,19,8),torch.randn(1,19,8)
        caches=expanded=None
        for rows in (torch.arange(19),torch.tensor([0,4,7,16]),torch.tensor([3,9])):
            q,k,v=torch.randn(1,6,len(rows),8),torch.randn(1,2,len(rows),8),torch.randn(1,2,len(rows),8)
            old=tuple(t.clone() for t in caches) if caches else None
            got,caches,expanded=prepare(q,k,v,rows,cos,sin,caches,expanded,19,rotate,repeat)
            refq,refk=rotate(q,k,cos[:,rows],sin[:,rows])
            torch.testing.assert_close(got,refq,rtol=0,atol=0)
            torch.testing.assert_close(caches[0][:,:,rows],refk,rtol=0,atol=0)
            torch.testing.assert_close(caches[1][:,:,rows],v,rtol=0,atol=0)
            for cache,mirror in zip(caches,expanded):torch.testing.assert_close(mirror,repeat(cache,3),rtol=0,atol=0)
            if old:
                untouched=torch.ones(19,dtype=torch.bool);untouched[rows]=False
                for a,b in zip(old,caches):torch.testing.assert_close(a[:,:,untouched],b[:,:,untouched],rtol=0,atol=0)

    def test_partial_initialization_refused(self):
        q,k,v=torch.randn(1,6,4,8),torch.randn(1,2,4,8),torch.randn(1,2,4,8)
        with self.assertRaisesRegex(RuntimeError,'every KV'):
            prepare(q,k,v,torch.arange(4),torch.ones(1,19,8),torch.zeros(1,19,8),None,None,19,rotate,repeat)


if __name__=='__main__':unittest.main()
