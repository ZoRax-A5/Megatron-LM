"""Distributed smoke/correctness checks for TP-synchronous proactive QKV replay."""

import copy
import os

import pytest
import torch
import torch.distributed as dist


def test_tp2_te_qkv_replay_matches():
    pytest.importorskip("transformer_engine")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("launch with exactly two ranks to exercise TP2")

    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import (
        get_recompute_checkpoint,
        model_parallel_cuda_manual_seed,
        pending_recompute_checkpoints,
        prefetch_recompute,
    )
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    model_parallel_cuda_manual_seed(
        431,
        tp_rank=dist.get_rank(),
        ep_rank=0,
        etp_rank=dist.get_rank(),
        force_reset_rng=True,
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=4,
        ffn_hidden_size=32,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        tensor_model_parallel_size=2,
        sequence_parallel=False,
        gradient_accumulation_fusion=True,
        transformer_impl="transformer_engine",
        init_method=init_method_normal(0.02),
        recompute_granularity="selective",
        recompute_modules=["attn_qkv"],
        proactive_recompute=True,
    )
    submodules = get_gpt_layer_with_transformer_engine_submodules(
        qk_layernorm=False
    ).self_attention.submodules
    process_groups = ProcessGroupCollection()
    process_groups.tp = dist.group.WORLD
    process_groups.cp = None
    template = SelfAttention(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=process_groups,
    ).cuda().to(torch.bfloat16).train()
    initial_state = copy.deepcopy(template.state_dict())
    input_value = torch.randn(4, 2, 16, device="cuda", dtype=torch.bfloat16)

    def run(mode):
        attention = SelfAttention(
            config=copy.deepcopy(config),
            submodules=submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=process_groups,
        ).cuda().to(torch.bfloat16).train()
        attention.load_state_dict(initial_state)
        attention.linear_qkv.is_first_microbatch = True
        attention.config.recompute_modules = [] if mode == "none" else ["attn_qkv"]
        attention.config.proactive_recompute = mode == "proactive"
        for parameter in attention.parameters():
            parameter.main_grad = torch.zeros_like(parameter)
            parameter.grad_added_to_main_grad = False
        hidden_states = input_value.detach().clone().requires_grad_(True)
        mixed_qkv, _ = attention.get_query_key_value_tensors(
            hidden_states, split_qkv=False
        )
        handle = None
        if mode == "proactive":
            checkpoint_id = pending_recompute_checkpoints()[-1]
            handle = get_recompute_checkpoint(checkpoint_id)
            assert prefetch_recompute(checkpoint_id, trigger_site="all_rank_tp_gate")
        loss = mixed_qkv.float().square().mean()
        loss.backward()
        effective_grads = []
        for parameter in attention.parameters():
            if getattr(parameter, "grad_added_to_main_grad", False):
                grad = parameter.main_grad
            elif parameter.grad is not None:
                grad = parameter.grad
            else:
                grad = parameter.main_grad
            effective_grads.append(grad.detach().clone())
            parameter.grad = grad
        torch.optim.SGD(attention.parameters(), lr=0.01).step()
        updated = [parameter.detach().clone() for parameter in attention.parameters()]
        return mixed_qkv.detach(), hidden_states.grad.detach(), effective_grads, updated, loss.detach(), handle

    reference = run("none")
    reference_repeat = run("none")
    on_demand = run("on_demand")
    proactive = run("proactive")
    for left, right in zip(reference[:5], reference_repeat[:5]):
        if isinstance(left, list):
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    for result in (on_demand, proactive):
        for index in (0, 1, 4):
            torch.testing.assert_close(result[index], reference[index], rtol=0, atol=0)
        for actual, expected in zip(result[2], reference[2]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(result[3], reference[3]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert proactive[5].state == "consumed"
    assert proactive[5]._cached_outputs is None
    dist.barrier()
    dist.destroy_process_group()


def test_tp2_dp4_te_qkv_three_step_distributed_optimizer_shards_match():
    pytest.importorskip("transformer_engine")
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        pytest.skip("launch with eight ranks to exercise TP2/DP4")

    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import (
        get_recompute_checkpoint,
        model_parallel_cuda_manual_seed,
        pending_recompute_checkpoints,
        prefetch_recompute,
        proactive_recompute_context,
        set_proactive_recompute_cache_limit,
    )
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world_rank = dist.get_rank()
    tp_rank = world_rank % 2
    dp_rank = world_rank // 2
    tp_groups = [
        dist.new_group(ranks=[2 * dp + tp for tp in range(2)]) for dp in range(4)
    ]
    dp_groups = [
        dist.new_group(ranks=[2 * dp + tp for dp in range(4)]) for tp in range(2)
    ]
    tp_group = tp_groups[dp_rank]
    dp_group = dp_groups[tp_rank]
    model_parallel_cuda_manual_seed(
        9917,
        tp_rank=tp_rank,
        ep_rank=0,
        etp_rank=tp_rank,
        force_reset_rng=True,
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=4,
        ffn_hidden_size=32,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        tensor_model_parallel_size=2,
        gradient_accumulation_fusion=True,
        transformer_impl="transformer_engine",
        init_method=init_method_normal(0.02),
        recompute_granularity="selective",
        recompute_modules=["attn_qkv"],
        proactive_recompute=True,
    )
    submodules = get_gpt_layer_with_transformer_engine_submodules(
        qk_layernorm=False
    ).self_attention.submodules
    pgs = ProcessGroupCollection()
    pgs.tp = tp_group
    pgs.dp = dp_group
    pgs.cp = None
    template = SelfAttention(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pgs,
    ).cuda().to(torch.bfloat16).train()
    initial_state = copy.deepcopy(template.state_dict())
    set_proactive_recompute_cache_limit(1)
    modes = {mode: None for mode in ("none", "on_demand", "proactive")}
    steps = 3

    def initialize_mode():
        module = SelfAttention(
            config=copy.deepcopy(config),
            submodules=submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pgs,
        ).cuda().to(torch.bfloat16).train()
        module.load_state_dict(initial_state)
        return module

    for mode in modes:
        module = initialize_mode()
        mode_results = []
        for step in range(steps):
            module.linear_qkv.is_first_microbatch = True
            module.config.recompute_modules = [] if mode == "none" else ["attn_qkv"]
            module.config.proactive_recompute = mode == "proactive"
            for parameter in module.parameters():
                parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
                parameter.grad_added_to_main_grad = False
                parameter.grad = None
            generator = torch.Generator(device="cuda").manual_seed(5000 + dp_rank + step)
            hidden = torch.randn(
                4, 2, 16, generator=generator, device="cuda", dtype=torch.bfloat16
            ).requires_grad_(True)
            with proactive_recompute_context(step=step, microbatch=0):
                mixed_qkv, _ = module.get_query_key_value_tensors(
                    hidden, split_qkv=False
                )
            handle = None
            if mode == "proactive":
                checkpoint_id = pending_recompute_checkpoints()[-1]
                handle = get_recompute_checkpoint(checkpoint_id)
                assert prefetch_recompute(
                    checkpoint_id, trigger_site="tp2_dp4_step_boundary"
                )
            loss = mixed_qkv.float().square().mean()
            loss.backward()
            effective_grads = []
            for parameter in module.parameters():
                if getattr(parameter, "grad_added_to_main_grad", False):
                    grad = parameter.main_grad
                elif parameter.grad is not None:
                    grad = parameter.grad
                else:
                    grad = parameter.main_grad
                effective_grads.append(grad.detach().float().clone())

            # Exercise the Distributed Optimizer's sharded-gradient/update shape:
            # reduce-scatter each gradient, update the DP-owned parameter shard,
            # then all-gather the updated parameter for the next step.
            shard_grads = []
            updated_shards = []
            dp_world = dist.get_world_size(dp_group)
            dp_local_rank = dist.get_rank(dp_group)
            lr = 0.003
            for parameter, grad in zip(module.parameters(), effective_grads):
                flat_grad = grad.reshape(-1).contiguous()
                shard_size = (flat_grad.numel() + dp_world - 1) // dp_world
                padded = torch.zeros(
                    shard_size * dp_world, device=grad.device, dtype=grad.dtype
                )
                padded[: flat_grad.numel()] = flat_grad
                reduced_shard = torch.empty(shard_size, device=grad.device, dtype=grad.dtype)
                dist.reduce_scatter_tensor(
                    reduced_shard, padded, op=dist.ReduceOp.SUM, group=dp_group
                )
                reduced_shard.div_(dp_world)
                shard_grads.append(reduced_shard.detach().clone())
                flat_parameter = parameter.detach().float().reshape(-1)
                start = dp_local_rank * shard_size
                local_parameter = torch.zeros_like(reduced_shard)
                local_count = max(0, min(shard_size, flat_parameter.numel() - start))
                if local_count:
                    local_parameter[:local_count] = flat_parameter[start : start + local_count]
                local_parameter.add_(reduced_shard, alpha=-lr)
                gathered = torch.empty(
                    dp_world * shard_size, device=grad.device, dtype=grad.dtype
                )
                dist.all_gather_into_tensor(gathered, local_parameter, group=dp_group)
                updated = gathered[: flat_parameter.numel()].to(parameter.dtype)
                parameter.data.copy_(updated.reshape_as(parameter))
                updated_shards.append(updated.detach().clone())
            mode_results.append(
                {
                    "loss": loss.detach().clone(),
                    "input_grad": hidden.grad.detach().clone(),
                    "grad_shards": shard_grads,
                    "parameters": [p.detach().clone() for p in module.parameters()],
                    "updated_shards": updated_shards,
                    "handle": handle,
                }
            )
            if handle is not None:
                assert handle.state == "consumed"
                assert handle._cached_outputs is None
        modes[mode] = mode_results

    reference = modes["none"]
    for mode in ("on_demand", "proactive"):
        for step in range(steps):
            expected = reference[step]
            actual = modes[mode][step]
            for field in ("loss", "input_grad"):
                torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)
            for field in ("grad_shards", "parameters", "updated_shards"):
                for actual_value, expected_value in zip(actual[field], expected[field]):
                    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    dist.barrier()
    set_proactive_recompute_cache_limit(None)
    dist.destroy_process_group()
