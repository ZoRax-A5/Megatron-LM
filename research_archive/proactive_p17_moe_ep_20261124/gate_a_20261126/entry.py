"""Explicit Gate A experiment entry. P17_MOE_PROACTIVE=1 opts in."""
import os,json,runpy
import megatron.training.training as training
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.tensor_parallel.random import ProactiveCheckpoint,assert_no_live_proactive_recompute_checkpoints
old_step=training.train_step
old_back=ProactiveCheckpoint._backward_replay

def back(self,*a,**k):
    try:return old_back(self,*a,**k)
    finally:
        with open(os.environ['P17_VERIFY_OUT']+'.lifecycle.rank'+os.environ['RANK']+'.jsonl','a') as f:
            f.write(json.dumps({'id':self.checkpoint_id,'state':self.state,'events':self.events,'cache_bytes':self.cache_bytes},default=str)+'\n')
ProactiveCheckpoint._backward_replay=back

def step(*a,**k):
    for model in a[2]:
        for m in model.modules():
            if isinstance(m,MoELayer):
                m.experts.p17_proactive_fc1=os.getenv('P17_MOE_PROACTIVE')=='1'
                m.experts.p17_layer=m.layer_number
    result=old_step(*a,**k)
    assert_no_live_proactive_recompute_checkpoints()
    return result
training.train_step=step
runpy.run_path('research_archive/proactive_p17_moe_ep_20261124/gate_a_20261126/correctness_entry.py' if os.getenv('P17_SNAPSHOT','1')=='1' else 'research_archive/proactive_p17_moe_ep_20261124/instrumented_entry.py',run_name='__main__')
