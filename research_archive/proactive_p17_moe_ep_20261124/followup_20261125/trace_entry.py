"""Explicit NVTX diagnostics. Adds identities; does not change communication order."""
import os,json,runpy,threading
import torch,torch.distributed as dist
import megatron.training.training as training
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.tensor_parallel.mappings import _AllToAll
def escape(value): return value.replace('{','{{').replace('}','}}')
s=threading.local(); rank=int(os.environ['RANK']); current={'step':None,'counts':{}}
orig_step=training.train_step

def step(*a,**k):
    current['step']=k.get('iteration'); current['counts']={}
    with torch.cuda.nvtx.range('P17STEP_'+str(current['step'])): return orig_step(*a,**k)
training.train_step=step
old_layer=MoELayer.forward

def layer(self,*a,**k):
    old=getattr(s,'meta',None); number=self.layer_number
    i=current['counts'].get(number,0); current['counts'][number]=i+1
    s.meta={'rank':rank,'step':current['step'],'layer':number,'microbatch':i}
    try:
        with torch.cuda.nvtx.range(escape('P17LAYER_'+json.dumps(s.meta,separators=(',',':')))): return old_layer(self,*a,**k)
    finally: s.meta=old
MoELayer.forward=layer
for name in ('token_dispatch','token_combine'):
    original=getattr(MoEAlltoAllTokenDispatcher,name)
    def wrap(fn,n):
        def call(self,*a,**k):
            old=getattr(s,'op',None);s.op=n
            try:return fn(self,*a,**k)
            finally:s.op=old
        return call
    setattr(MoEAlltoAllTokenDispatcher,name,wrap(original,name))
from megatron.core.tensor_parallel.random import CheckpointFunction
old_checkpoint_backward=CheckpointFunction.backward
def checkpoint_backward(ctx,*a):
    with torch.cuda.nvtx.range('P17_CHECKPOINT_BACKWARD'):
        return old_checkpoint_backward(ctx,*a)
CheckpointFunction.backward=staticmethod(checkpoint_backward)
old_fwd,old_bwd=_AllToAll.forward,_AllToAll.backward

def fwd(ctx,group,tensor,out_splits,in_splits,*a):
    def simple(x): return x.tolist() if hasattr(x,'tolist') else x
    meta=dict(getattr(s,'meta',None) or {});meta.update(op=getattr(s,'op',None),phase='forward',shape=list(tensor.shape),dtype=str(tensor.dtype),group=dist.get_process_group_ranks(group),out_splits=simple(out_splits),in_splits=simple(in_splits))
    ctx.followup_meta=meta
    with torch.cuda.nvtx.range(escape('P17META_'+json.dumps(meta,separators=(',',':')))): return old_fwd(ctx,group,tensor,out_splits,in_splits,*a)

def bwd(ctx,*a):
    meta=dict(ctx.followup_meta);meta['phase']='backward'
    with torch.cuda.nvtx.range(escape('P17META_'+json.dumps(meta,separators=(',',':')))): return old_bwd(ctx,*a)
_AllToAll.forward=staticmethod(fwd);_AllToAll.backward=staticmethod(bwd)
runpy.run_path('research_archive/proactive_p17_moe_ep_20261124/instrumented_entry.py',run_name='__main__')
