"""Observational FC1 call counting; wrapped after parameter discovery, no new replay."""
import os,json,runpy,weakref,threading
from contextlib import nullcontext
import torch
from megatron.core.tensor_parallel.random import ProactiveCheckpoint,CheckpointFunction,get_proactive_recompute_context
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
consumers=[]
old_combine=MoEAlltoAllTokenDispatcher.token_combine;old_postprocess=MoEAlltoAllTokenDispatcher.combine_postprocess

def observe_combine(self,*a,**k):
    result=old_combine(self,*a,**k)
    if getattr(self,'p17_target',False):
        self.p17_output_ref=weakref.ref(result)
        self.p17_output_ptr=result.data_ptr()
    return result

def postprocess(self,states,*a,**k):
    target=getattr(self,'p17_target',False);context=get_proactive_recompute_context()
    if target:
        consumers.append({'rank':int(os.environ['RANK']),'step':context.get('step'),'microbatch':context.get('microbatch'),'same_object':self.p17_output_ref() is states,'same_storage_pointer':self.p17_output_ptr==states.data_ptr(),'shape':list(states.shape)})
    scope=torch.cuda.nvtx.range('P17_TARGET_CONSUMER_'+str(context.get('step'))) if target else nullcontext()
    with scope:return old_postprocess(self,states,*a,**k)
MoEAlltoAllTokenDispatcher.token_combine=observe_combine
MoEAlltoAllTokenDispatcher.combine_postprocess=postprocess

tls=threading.local()
old_expert=TEGroupedMLP.forward;old_native_forward=CheckpointFunction.forward;old_native_backward=CheckpointFunction.backward

def expert(self,*a,**k):
    previous=getattr(tls,'layer',None);tls.layer=getattr(self,'p17_layer',None)
    try:return old_expert(self,*a,**k)
    finally:tls.layer=previous
TEGroupedMLP.forward=expert

def native_forward(ctx,*a):
    context=get_proactive_recompute_context();ctx.p17_target=getattr(tls,'layer',None)==1 and context.get('microbatch')==0;ctx.p17_step=context.get('step')
    return old_native_forward(ctx,*a)

def native_backward(ctx,*a):
    scope=torch.cuda.nvtx.range('P17_NATIVE_BACKWARD_'+str(ctx.p17_step)) if ctx.p17_target else nullcontext()
    with scope:return old_native_backward(ctx,*a)
CheckpointFunction.forward=staticmethod(native_forward);CheckpointFunction.backward=staticmethod(native_backward)
old_init=ProactiveCheckpoint.__init__;old_backward=ProactiveCheckpoint._backward_replay;rows=[]
def init(self,*a,**k):
    old_init(self,*a,**k)
    fn=self.run_function;ref=weakref.ref(self);counts={};self.p17_calls=counts;descriptors=[];self.p17_descriptors=descriptors
    def counted(*inputs):
        handle=ref();phase='forward' if not torch.is_grad_enabled() else 'prefetch' if handle.state=='prefetch_submitted' else 'backward'
        counts[phase]=counts.get(phase,0)+1
        descriptors.append({'phase':phase,'inputs':[{'pointer':t.data_ptr(),'shape':list(t.shape),'version':t._version} for t in inputs],'parameter_versions':[p._version for p in handle.parameters],'rng_saved':handle._rng_states is not None})
        with torch.cuda.nvtx.range(f'P17_FC1_CALL_{phase}_{handle.checkpoint_id}'):return fn(*inputs)
    self.run_function=counted
ProactiveCheckpoint.__init__=init

def backward(self,*a,**k):
    versions=[p._version for p in self.parameters]
    graph_present=self._cached_outputs is not None and any(t is not None and t.grad_fn is not None for t in self._cached_outputs)
    try:
        with torch.cuda.nvtx.range(f'P17_ACTIVE_BACKWARD_{self.checkpoint_id}'):return old_backward(self,*a,**k)
    finally:rows.append({'id':self.checkpoint_id,'metadata':self.metadata,'calls':self.p17_calls,'descriptors':self.p17_descriptors,'parameter_versions_at_consume':versions,'cached_grad_fn_at_consume':graph_present,'state':self.state,'rank':int(os.environ['RANK'])})
ProactiveCheckpoint._backward_replay=backward
try:
    runpy.run_path('research_archive/proactive_p17_moe_ep_20261124/gate_b_20261126/entry_serial_control.py',run_name='__main__')
finally:
    with open(os.environ['P17_OUTPUT']+'.calls.rank'+os.environ['RANK']+'.jsonl','w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')

    with open(os.environ['P17_OUTPUT']+'.consumers.rank'+os.environ['RANK']+'.jsonl','w') as f:
        for row in consumers:f.write(json.dumps(row)+'\n')
