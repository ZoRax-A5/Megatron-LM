"""Opt-in two-microbatch schedule experiment using actual TE QKV."""
import copy
import json
import os
import time

import pytest
import torch


def test_cross_microbatch_te_schedule():
    pytest.importorskip('transformer_engine')
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_submodules
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import (model_parallel_cuda_manual_seed, proactive_recompute_context,
        pending_recompute_checkpoints, get_recompute_checkpoint, prefetch_recompute,
        assert_no_live_proactive_recompute_checkpoints, set_proactive_recompute_cache_limit)
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal

    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234, tp_rank=0, ep_rank=0, etp_rank=0, force_reset_rng=True)
    config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4,
        num_query_groups=4, kv_channels=16, ffn_hidden_size=128, params_dtype=torch.bfloat16,
        normalization='RMSNorm', tensor_model_parallel_size=1, gradient_accumulation_fusion=True,
        transformer_impl='transformer_engine', init_method=init_method_normal(0.02),
        recompute_granularity='selective', recompute_modules=['attn_qkv'], proactive_recompute=True)
    pg = ProcessGroupCollection(); pg.tp = None; pg.cp = None
    sub = get_gpt_layer_with_transformer_engine_submodules(qk_layernorm=False).self_attention.submodules
    template = SelfAttention(config=config, submodules=sub, layer_number=1,
        attn_mask_type=AttnMaskType.causal, pg_collection=pg).cuda().to(torch.bfloat16).train()
    assert isinstance(template.linear_qkv, TELayerNormColumnParallelLinear)
    initial = copy.deepcopy(template.state_dict())
    inputs = [torch.randn(64, 2, 64, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    set_proactive_recompute_cache_limit(1)

    def run(mode):
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m = copy.deepcopy(template); m.load_state_dict(initial)
        m.config.proactive_recompute = mode == 'proactive'
        for p in m.parameters():
            p.main_grad = torch.zeros_like(p, dtype=torch.float32); p.grad_added_to_main_grad = False
        xs = [x.detach().clone().requires_grad_() for x in inputs]
        losses = [None, None]; handles = {}; counts = {'forward': 0}; marks = {}; cpu = {}; guard = [False]
        def mark(name):
            cpu[name] = time.perf_counter(); marks[name] = torch.cuda.Event(enable_timing=True); marks[name].record()
        def hook(mod, args):
            counts['forward'] += 1
            counts[stage[0]] = counts.get(stage[0], 0) + 1
            if mode == 'proactive' and stage[0] == 'B' and not guard[0]:
                guard[0] = True
                stage[0] = 'replay_A'
                assert handles[0].metadata['microbatch'] == 0
                assert handles[0].metadata['step'] == 7
                mark('replay_start'); torch.cuda.nvtx.range_push('B_hook_replay_A')
                assert prefetch_recompute(handles[0].checkpoint_id, trigger_site='B_qkv_pre_hook')
                torch.cuda.nvtx.range_pop(); mark('replay_end'); stage[0] = 'B'
        m.linear_qkv.register_forward_pre_hook(hook)
        stage = ['']
        def forward(i):
            stage[0] = 'AB'[i]; mark('forward_'+stage[0]+'_start'); torch.cuda.nvtx.range_push('forward_'+stage[0])
            m.linear_qkv.is_first_microbatch = (i == 0)
            with proactive_recompute_context(step=7, microbatch=i):
                out, _ = m.get_query_key_value_tensors(xs[i], split_qkv=False)
            losses[i] = out.float().square().mean() / 2
            if mode == 'proactive':
                ids = [x for x in pending_recompute_checkpoints() if get_recompute_checkpoint(x).metadata.get('microbatch') == i]
                assert len(ids) == 1; handles[i] = get_recompute_checkpoint(ids[0])
            torch.cuda.nvtx.range_pop(); mark('forward_'+stage[0]+'_end')
        def backward(i):
            stage[0] = 'backward_'+str(i); mark(stage[0]+'_start'); losses[i].backward(); mark(stage[0]+'_end')
        forward(0)
        if mode == 'original': backward(0)
        forward(1)
        if mode != 'original': backward(0)
        backward(1)
        grads=[]
        for p in m.parameters():
            g = p.main_grad if p.grad_added_to_main_grad or p.grad is None else p.grad
            grads.append(g.detach().clone()); p.grad = g.to(p.dtype)
        torch.optim.SGD(m.parameters(), lr=0.01).step()
        params=[p.detach().clone() for p in m.parameters()]
        torch.cuda.synchronize()
        metrics={'peak_allocated':torch.cuda.max_memory_allocated(), 'peak_reserved':torch.cuda.max_memory_reserved(),
                 'cpu':cpu, 'counts':counts, 'gpu_ms':{k:marks['forward_A_start'].elapsed_time(v) for k,v in marks.items()}}
        if mode == 'proactive':
            h = handles[0]; names=[e['event'] for e in h.events]
            assert names.count('prefetch_complete') == names.count('cache_hit') == names.count('cache_consumed') == names.count('released') == 1
            assert 'fallback' not in names and h.state == 'consumed' and h._cached_outputs is None
            assert counts == {'forward': 4, 'A': 1, 'B': 1, 'replay_A': 1, 'backward_1': 1}
            assert cpu['forward_B_start'] < cpu['replay_start'] < cpu['replay_end'] < cpu['forward_B_end'] < cpu['backward_0_start']
            g=metrics['gpu_ms']; assert g['forward_B_start'] <= g['replay_start'] <= g['replay_end'] <= g['forward_B_end'] <= g['backward_0_start']
            complete = next(e for e in h.events if e['event'] == 'prefetch_complete')
            metrics.update(cache_bytes=complete['output_bytes'], saved_tensor_bytes=complete['saved_tensor_bytes'], memory=h.memory_snapshot, lifetime_s=h.cache_lifetime_seconds, events=h.events)
            assert_no_live_proactive_recompute_checkpoints()
        return [x.detach().cpu() for x in losses], [x.grad.cpu() for x in xs], [g.cpu() for g in grads], [p.cpu() for p in params], metrics

    baseline=run('original'); repeat=run('original'); adjusted=run('adjusted'); proactive=run('proactive')
    for candidate in (repeat, adjusted, proactive):
        for ref_group, actual_group in zip(baseline[:4], candidate[:4]):
            for ref, actual in zip(ref_group, actual_group): torch.testing.assert_close(ref, actual, rtol=0, atol=0)
    report={'original':baseline[4], 'baseline_repeat':repeat[4], 'adjusted_on_demand':adjusted[4], 'adjusted_proactive':proactive[4], 'exact_equal':True}
    path=os.environ.get('P8_RESULT_JSON')
    if path:
        with open(path,'w') as f: json.dump(report,f,indent=2)
    print('P8_RESULT',json.dumps(report))
