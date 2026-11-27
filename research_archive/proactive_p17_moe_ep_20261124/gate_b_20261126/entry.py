"""Fixed single-instance P17 PoC, default C0. Explicit experiment entry only."""
import os,json,runpy,logging
from contextlib import nullcontext
import torch
import megatron.training.training as training
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.tensor_parallel.random import ProactiveCheckpoint,proactive_recompute_context,assert_no_live_proactive_recompute_checkpoints
if os.getenv('P17_QUIET')=='1':logging.getLogger('megatron.core.tensor_parallel.random').setLevel(logging.WARNING)
lifecycle=[]
mode=os.getenv('P17_MODE','C0'); pending={}; counters={}; active={}; replay=None
old_step=training.train_step;old_layer=MoELayer.forward;old_combine=MoEAlltoAllTokenDispatcher.token_combine;old_backward=ProactiveCheckpoint._backward_replay

def schedule(handle):
    ready=torch.cuda.Event();ready.record();handle.p17_ready=ready
    if pending:raise RuntimeError('stale P17 handle')
    pending['handle']=handle
    if mode=='C1':handle.prefetch('P17_C1_safe')

def layer(self,*a,**k):
    mb=counters.get(self.layer_number,0);counters[self.layer_number]=mb+1
    self.experts.p17_proactive_fc1=mode!='C0' and self.layer_number==1 and mb==0
    self.experts.p17_layer=self.layer_number;self.experts.p17_fc1_prefetch=schedule
    self.token_dispatcher.p17_target=self.layer_number==3 and mb==0
    with proactive_recompute_context(step=active['step'],microbatch=mb):return old_layer(self,*a,**k)
MoELayer.forward=layer

def combine(self,*a,**k):
    global replay
    scope=torch.cuda.nvtx.range('P17_TARGET_COMBINE_'+str(active['step'])) if getattr(self,'p17_target',False) else nullcontext()
    with scope:output=old_combine(self,*a,**k)
    if getattr(self,'p17_target',False) and mode=='C2':
        handle=pending['handle']
        if replay is None:replay=torch.cuda.Stream()
        replay.wait_event(handle.p17_ready)
        with torch.cuda.stream(replay):
            for tensor in handle._inputs:tensor.record_stream(replay)
            handle.prefetch('P17_C2_layer3_combine')
            handle.p17_done=torch.cuda.Event();handle.p17_done.record()
    return output
MoEAlltoAllTokenDispatcher.token_combine=combine

def backward(self,*a,**k):
    if self.metadata.get('module')!='moe_fc1':return old_backward(self,*a,**k)
    current=torch.cuda.current_stream()
    if mode=='C2':
        current.wait_event(self.p17_done)
        replay.wait_stream(current)
        for grad in a[0]:
            if torch.is_tensor(grad):grad.record_stream(replay)
    try:return old_backward(self,*a,**k)
    finally:
        if mode=='C2':current.wait_stream(replay)
        lifecycle.append({'id':self.checkpoint_id,'state':self.state,'events':self.events,'cache_bytes':self.cache_bytes,'graph_released':self._cached_outputs is None and self._cached_inputs is None and self._inputs is None})
        if pending.pop('handle',None) is not self:raise RuntimeError('P17 identity mismatch')
ProactiveCheckpoint._backward_replay=backward

def step(*a,**k):
    if pending:raise RuntimeError('cross-step pending handle')
    counters.clear();active['step']=k['iteration']
    result=old_step(*a,**k)
    if pending:raise RuntimeError('unconsumed handle')
    assert_no_live_proactive_recompute_checkpoints()
    return result
training.train_step=step
try:
    runpy.run_path('research_archive/proactive_p17_moe_ep_20261124/gate_a_20261126/correctness_entry.py' if os.getenv('P17_SNAPSHOT','1')=='1' else 'research_archive/proactive_p17_moe_ep_20261124/instrumented_entry.py',run_name='__main__')
finally:
    with open(os.environ['P17_VERIFY_OUT']+'.lifecycle.rank'+os.environ['RANK']+'.jsonl','a') as f:
        for row in lifecycle:f.write(json.dumps(row,default=str)+'\n')
