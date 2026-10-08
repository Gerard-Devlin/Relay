import unittest
from types import SimpleNamespace as NS
import torch
from relay_cache.cache import Engine,RelayFrontier


def fused_cache_attention(block,x,layer,positions,lengths,softmax_scale=None):
    block.k_cache.index_copy_(0,positions[0].long(),x)
    return (x+1).unsqueeze(0)

class Block:
    def __init__(self):self.k_cache=torch.zeros((8,2),dtype=torch.float64)

def model():
    return NS(config=NS(n_layers=32,d_model=2),dtype=torch.float64,device=torch.device('cpu'),
        model=NS(transformer=NS(blocks=[Block() for _ in range(32)])))

def call(engine,frontier,labels,optional=()):
    q=torch.arange(8);ids=torch.tensor(labels,dtype=torch.long)
    frontier.optional=torch.tensor(optional,dtype=torch.long)
    tables=torch.tensor([[0,8,0,2]],dtype=torch.int32)
    tracked=torch.tensor([[0,8,2,8]],dtype=torch.int32)
    positions=[q,torch.empty(0,dtype=torch.long),[],[],torch.zeros(8),[]]
    lengths=[[],None,tables,tracked,torch.cat((tables,tracked)),[0],1,8,32,128,None,False]
    engine.begin(ids,positions,lengths,frontier,2)
    x=ids[:,None].double().expand(8,2).clone()
    for layer,block in enumerate(engine.model.model.transformer.blocks):
        x=engine.forward(block,x,layer,positions,lengths).squeeze(0)
    engine.end();return x


class TestRuntime(unittest.TestCase):
    def test_full_wrapper_is_exact_in_toy_program(self):
        m=model();f=RelayFrontier();f.reset(8,'cpu');e=Engine(m,True,True)
        labels=list(range(8))
        for _ in range(5):
            got=call(e,f,labels,(2,3,4))
            self.assertTrue(torch.equal(got,torch.tensor(labels)[:,None].expand(8,2)+32))
    def test_actual_projection_row_ledger(self):
        m=model();f=RelayFrontier();f.reset(8,'cpu');e=Engine(m,True)
        call(e,f,list(range(8)))
        before=e.row_layers;call(e,f,list(range(8)),(2,3,4,5))
        self.assertEqual(e.row_layers-before,4*32+4*8)
        self.assertEqual(e.optional_skipped_row_layers,4*24)
    def test_dirty_tokens_run_all_layers(self):
        m=model();f=RelayFrontier();f.reset(8,'cpu');e=Engine(m,True)
        call(e,f,list(range(8)));f.dirty[7]=True
        got=call(e,f,[0,1,2,3,4,5,6,100],(2,3,4,5))
        self.assertTrue(torch.equal(got[7],torch.tensor([132.,132.],dtype=torch.float64)))
    def test_identity_guard_detects_wrong_reuse(self):
        m=model();f=RelayFrontier();f.reset(8,'cpu');e=Engine(m,True)
        call(e,f,list(range(8)))
        with self.assertRaises(AssertionError):call(e,f,[0,1,100,3,4,5,6,7],(2,))
    def test_module_restored_on_exception(self):
        m=model();e=Engine(m,True);old=globals()['fused_cache_attention']
        with self.assertRaises(ValueError):
            with e.installed():raise ValueError('test cleanup')
        self.assertIs(globals()['fused_cache_attention'],old)

if __name__=='__main__':unittest.main()
