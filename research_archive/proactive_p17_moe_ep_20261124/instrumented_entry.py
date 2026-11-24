"""Optional P17 diagnostics; invoked explicitly, leaves normal Megatron paths unchanged."""
import json, os, runpy
import torch
import torch.distributed as dist
import megatron.training.training as training
from megatron.core import parallel_state as ps
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.router import TopKRouter

path = os.environ['P17_OUTPUT']
rank = int(os.environ['RANK'])
def record(obj):
    with open(f'{path}.rank{rank}.jsonl', 'a') as f: f.write(json.dumps(obj)+'\n')
orig_step = training.train_step
first = True
def step(*a, **kw):
    global first
    if first:
        groups = {name: dist.get_process_group_ranks(getter()) for name, getter in {
            'tp': ps.get_tensor_model_parallel_group, 'pp': ps.get_pipeline_model_parallel_group,
            'cp': ps.get_context_parallel_group, 'dense_dp': ps.get_data_parallel_group,
            'ep': ps.get_expert_model_parallel_group, 'expert_dp': ps.get_expert_data_parallel_group}.items()}
        record({'groups':groups, 'moe_classes': sorted({type(m).__name__ for model in a[2] for m in model.modules() if isinstance(m, TEGroupedMLP)}), 'dispatchers': sorted({type(m.token_dispatcher).__name__ for model in a[2] for m in model.modules() if hasattr(m, 'token_dispatcher')})})
        torch.cuda.reset_peak_memory_stats()
        first = False
    optimizer = a[3]
    original_optimizer_step = optimizer.step
    if os.getenv('P17_SNAPSHOT') == '1' and kw.get('iteration', 0) == 0:
        def snapshot_step(*args, **kwargs):
            model = a[2]
            params = {f'{i}:{name}': p for i, module in enumerate(model) for name, p in module.named_parameters() if p.requires_grad}
            grads = {name: getattr(p, 'main_grad', None).detach().cpu().clone() for name, p in params.items() if getattr(p, 'main_grad', None) is not None}
            result = original_optimizer_step(*args, **kwargs)
            weights = {name: p.detach().cpu().clone() for name, p in params.items()}
            torch.save({'gradients': grads, 'parameters': weights, 'missing_gradients': sorted(set(params)-set(grads))}, f'{path}.rank{rank}.pt')
            return result
        optimizer.step = snapshot_step
    try:
        result = orig_step(*a, **kw)
    finally:
        optimizer.step = original_optimizer_step
    record({'iteration':kw.get('iteration'), 'peak_allocated':torch.cuda.max_memory_allocated(), 'peak_reserved':torch.cuda.max_memory_reserved()})
    return result
training.train_step = step
if os.getenv('P17_TRACE') == '1':
    from megatron.core.tensor_parallel.random import CheckpointFunction
    old_checkpoint_forward = CheckpointFunction.forward
    def checkpoint_forward(ctx, fn, distribute, *args):
        def traced_fn(*inputs):
            phase = 'replay' if torch.is_grad_enabled() else 'forward'
            with torch.cuda.nvtx.range('P17_checkpoint_' + phase):
                return fn(*inputs)
        return old_checkpoint_forward(ctx, traced_fn, distribute, *args)
    CheckpointFunction.forward = staticmethod(checkpoint_forward)
    import threading
    from megatron.core.tensor_parallel.mappings import _AllToAll
    state = threading.local()
    orig_a2a_forward, orig_a2a_backward = _AllToAll.forward, _AllToAll.backward
    def a2a_forward(ctx, group, tensor, *args):
        ctx.p17_label = getattr(state, 'label', 'unknown')
        with torch.cuda.nvtx.range('P17_A2A_fwd_' + ctx.p17_label + '_shape_' + str(tuple(tensor.shape))):
            return orig_a2a_forward(ctx, group, tensor, *args)
    def a2a_backward(ctx, *args):
        with torch.cuda.nvtx.range('P17_A2A_bwd_' + ctx.p17_label):
            return orig_a2a_backward(ctx, *args)
    _AllToAll.forward = staticmethod(a2a_forward)
    _AllToAll.backward = staticmethod(a2a_backward)
    def wrap(cls, name, label):
        orig = getattr(cls, name)
        def method(self, *args, **kwargs):
            previous = getattr(state, 'label', 'unknown')
            state.label = label
            try:
                with torch.cuda.nvtx.range(label):
                    return orig(self, *args, **kwargs)
            finally:
                state.label = previous
        setattr(cls, name, method)
    wrap(MoEAlltoAllTokenDispatcher, 'token_dispatch', 'P17_dispatch')
    wrap(MoEAlltoAllTokenDispatcher, 'token_combine', 'P17_combine')
    wrap(TEGroupedMLP, 'forward', 'P17_experts')
    wrap(TopKRouter, 'forward', 'P17_router')
if os.getenv('P17_ROUTING') == '1':
    orig_router = TopKRouter.forward
    def router(self, *a, **kw):
        probs, mapping = orig_router(self, *a, **kw)
        record({'layer':self.layer_number, 'counts':mapping.sum(dim=0).cpu().tolist(), 'tokens':mapping.shape[0]})
        return probs, mapping
    TopKRouter.forward = router
runpy.run_path('pretrain_gpt.py', run_name='__main__')
