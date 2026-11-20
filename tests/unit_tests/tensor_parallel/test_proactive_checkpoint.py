"""Single-device correctness tests for the experimental proactive checkpoint API."""

import copy
import pytest
import torch

from megatron.core.tensor_parallel.random import (
    ProactiveCheckpoint,
    checkpoint,
    assert_no_live_proactive_recompute_checkpoints,
    get_proactive_recompute_context,
    pending_recompute_checkpoints,
    prefetch_recompute,
    proactive_recompute_cache_stats,
    proactive_recompute_context,
    set_proactive_recompute_cache_limit,
)


def test_proactive_checkpoint_reuses_graph_and_matches_checkpoint(tmp_path):
    torch.manual_seed(123)
    module = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)
    ).cuda()
    reference = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)
    ).cuda()
    reference.load_state_dict(module.state_dict())
    baseline = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)
    ).cuda()
    baseline.load_state_dict(module.state_dict())
    x = torch.randn(3, 8, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    x_baseline = x.detach().clone().requires_grad_(True)
    calls = 0

    def run(inp):
        nonlocal calls
        calls += 1
        return module(inp)

    handle = ProactiveCheckpoint(run, parameters=module.parameters())
    output = handle(x)
    assert handle.prefetch()
    assert calls == 2  # original forward plus exactly one proactive replay
    assert handle.state == "prefetch_complete"
    assert handle.cache_bytes == output.numel() * output.element_size()
    completed = next(e for e in handle.events if e["event"] == "prefetch_complete")
    assert completed["saved_tensor_bytes"] > 0
    assert completed["allocated_bytes"] is not None
    assert completed["reserved_bytes"] is not None
    output.square().sum().backward()
    assert calls == 2  # backward consumed the graph without another replay

    baseline_output = baseline(x_baseline)
    baseline_output.square().sum().backward()
    expected = checkpoint(lambda inp: reference(inp), False, x_ref)
    expected.square().sum().backward()
    torch.testing.assert_close(output, baseline_output)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(x.grad, x_baseline.grad)
    for actual_param, expected_param in zip(
        module.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(actual_param.grad, expected_param.grad)
    for actual_param, baseline_param in zip(module.parameters(), baseline.parameters()):
        torch.testing.assert_close(actual_param.grad, baseline_param.grad)
    torch.optim.SGD(module.parameters(), lr=0.01).step()
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    torch.optim.SGD(baseline.parameters(), lr=0.01).step()
    for actual_param, expected_param in zip(
        module.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(actual_param, expected_param)
    for actual_param, baseline_param in zip(module.parameters(), baseline.parameters()):
        torch.testing.assert_close(actual_param, baseline_param)

    events = [entry["event"] for entry in handle.events]
    assert events.count("prefetch_start") == 1
    assert events.count("prefetch_complete") == 1
    assert events.count("cache_hit") == 1
    assert events.count("cache_consumed") == 1
    assert events.count("released") == 1
    assert handle.state == "consumed"
    assert handle._cached_outputs is None
    event_path = tmp_path / "events.json"
    handle.write_events(event_path)
    assert '"cache_hit"' in event_path.read_text()


def test_proactive_checkpoint_falls_back_when_not_prefetched():
    calls = 0

    def run(inp):
        nonlocal calls
        calls += 1
        return inp * 3

    x = torch.randn(4, device="cuda", requires_grad=True)
    handle = ProactiveCheckpoint(run)
    output = handle(x)
    output.sum().backward()
    assert calls == 2  # no-grad forward, then the original on-demand replay
    torch.testing.assert_close(x.grad, torch.full_like(x, 3))
    assert any(e["event"] == "fallback" for e in handle.events)
    assert handle._cached_outputs is None


def test_proactive_checkpoint_handles_do_not_mix_invocations():
    module = torch.nn.Linear(5, 3).cuda()
    inputs = [torch.randn(2, 5, device="cuda", requires_grad=True) for _ in range(2)]
    handles = [
        ProactiveCheckpoint(module, parameters=module.parameters()) for _ in inputs
    ]
    outputs = [handle(inp) for handle, inp in zip(handles, inputs)]
    assert handles[0].checkpoint_id != handles[1].checkpoint_id
    assert all(handle.prefetch() for handle in handles)

    outputs[1].sum().backward()
    outputs[0].sum().backward()

    for inp in inputs:
        torch.testing.assert_close(
            inp.grad, module.weight.detach().sum(dim=0).expand_as(inp)
        )
    assert all(handle.state == "consumed" for handle in handles)
    assert all(
        sum(e["event"] == "cache_hit" for e in handle.events) == 1 for handle in handles
    )


def test_proactive_checkpoint_checks_parameter_versions():
    module = torch.nn.Linear(4, 3).cuda()
    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    handle = ProactiveCheckpoint(module)
    output = handle(x)
    with torch.no_grad():
        module.weight.add_(1)
    with pytest.raises(RuntimeError, match="parameter changed"):
        handle.prefetch()
    with pytest.raises(RuntimeError):
        output.sum().backward()
    assert handle.state == "failed_terminal"
    assert module.weight.grad is None


def test_proactive_checkpoint_rejects_input_mutation_after_prefetch():
    x = torch.randn(4, device="cuda", requires_grad=True)
    handle = ProactiveCheckpoint(lambda value: value.sin())
    output = handle(x)
    assert handle.prefetch()
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError):
        output.sum().backward()
    assert handle.state == "failed_terminal"
    assert handle._cached_outputs is None


def test_proactive_checkpoint_rejects_parameter_mutation_after_prefetch():
    module = torch.nn.Linear(4, 2).cuda()
    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    handle = ProactiveCheckpoint(module)
    output = handle(x)
    assert handle.prefetch()
    with torch.no_grad():
        module.weight.add_(1)
    with pytest.raises(RuntimeError):
        output.sum().backward()
    assert handle.state == "failed_terminal"
    assert handle._cached_outputs is None
    assert module.weight.grad is None


def test_proactive_checkpoint_prefetch_exception_is_terminal_by_default():
    module = torch.nn.Linear(4, 2).cuda()
    fail_prefetch = False

    def run(value):
        if fail_prefetch and torch.is_grad_enabled():
            raise ValueError("injected replay error")
        return module(value)

    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    handle = ProactiveCheckpoint(run, parameters=module.parameters())
    output = handle(x)
    fail_prefetch = True
    with pytest.raises(RuntimeError, match="not certified safe"):
        handle.prefetch()
    fail_prefetch = False
    with pytest.raises(RuntimeError, match="failed_terminal"):
        output.sum().backward()
    assert module.weight.grad is None


def test_proactive_checkpoint_allows_opted_in_safe_prefetch_fallback():
    module = torch.nn.Linear(4, 2).cuda()
    fail_prefetch = False

    def run(value):
        if fail_prefetch and torch.is_grad_enabled():
            raise ValueError("injected replay error")
        return module(value)

    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    handle = ProactiveCheckpoint(
        run, parameters=module.parameters(), allow_prefetch_fallback=True
    )
    output = handle(x)
    fail_prefetch = True
    assert not handle.prefetch()
    fail_prefetch = False
    output.sum().backward()
    assert handle.state == "consumed"
    assert any(event["event"] == "fallback" for event in handle.events)
    assert module.weight.grad is not None


def test_proactive_checkpoint_dropout_rng_matches_on_demand():
    dropout = torch.nn.Dropout(p=0.4).cuda().train()
    x = torch.randn(5, 7, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    torch.manual_seed(991)
    torch.cuda.manual_seed_all(991)
    handle = ProactiveCheckpoint(dropout)
    output = handle(x)
    assert handle.prefetch()
    output.sum().backward()

    torch.manual_seed(991)
    torch.cuda.manual_seed_all(991)
    expected = checkpoint(dropout, False, x_ref)
    expected.sum().backward()
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(x.grad, x_ref.grad)


def test_proactive_checkpoint_supports_partial_multi_output_and_frozen_input():
    module = torch.nn.Linear(4, 3).cuda()
    x = torch.randn(2, 4, device="cuda", requires_grad=False)

    def run(value):
        first = module(value)
        return first, first * 2

    handle = ProactiveCheckpoint(run, parameters=module.parameters())
    first, second = handle(x)
    assert handle.prefetch()
    first.sum().backward()
    assert x.grad is None
    assert module.weight.grad is not None
    assert handle.state == "consumed"


def test_proactive_checkpoint_duplicate_prefetch_and_backward_are_rejected():
    x = torch.randn(3, device="cuda", requires_grad=True)
    handle = ProactiveCheckpoint(lambda value: value * value)
    output = handle(x)
    assert handle.prefetch()
    with pytest.raises(RuntimeError, match="prefetch_complete"):
        handle.prefetch()
    output.sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="already consumed"):
        output.sum().backward()


def test_proactive_te_qkv_matches_on_demand_and_no_recompute():
    """Exercise the actual SelfAttention QKV branch with Megatron's fused TE module."""
    pytest.importorskip("transformer_engine")
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import prefetch_recompute
    from megatron.core.tensor_parallel.random import get_recompute_checkpoint
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal

    torch.manual_seed(7193)
    torch.cuda.manual_seed_all(7193)
    model_parallel_cuda_manual_seed(7193, tp_rank=0, ep_rank=0, etp_rank=0, force_reset_rng=True)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=4,
        ffn_hidden_size=32,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        tensor_model_parallel_size=1,
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
    process_groups.tp = None
    process_groups.cp = None
    template = SelfAttention(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=process_groups,
    ).cuda().to(torch.bfloat16).train()
    assert isinstance(template.linear_qkv, TELayerNormColumnParallelLinear)
    initial_state = copy.deepcopy(template.state_dict())
    input_value = torch.randn(8, 2, 16, device="cuda", dtype=torch.bfloat16)

    def run(mode):
        attention = copy.deepcopy(template)
        attention.load_state_dict(initial_state)
        attention.linear_qkv.is_first_microbatch = True
        for parameter in attention.parameters():
            parameter.main_grad = torch.zeros_like(parameter)
            parameter.grad_added_to_main_grad = False
        if mode == "no_recompute":
            attention.config.recompute_modules = []
            attention.config.proactive_recompute = False
        else:
            attention.config.recompute_modules = ["attn_qkv"]
            attention.config.proactive_recompute = mode == "proactive"
        forward_count = [0]
        attention.linear_qkv.register_forward_hook(
            lambda *_: forward_count.__setitem__(0, forward_count[0] + 1)
        )
        hidden_states = input_value.detach().clone().requires_grad_(True)
        mixed_qkv, _ = attention.get_query_key_value_tensors(
            hidden_states, split_qkv=False
        )
        handle = None
        if mode == "proactive":
            checkpoint_id = pending_recompute_checkpoints()[0]
            handle = get_recompute_checkpoint(checkpoint_id)
            assert prefetch_recompute(checkpoint_id)
            assert handle.state == "prefetch_complete"
        loss = mixed_qkv.float().square().mean()
        loss.backward()
        grads = []
        for parameter in attention.parameters():
            if getattr(parameter, "grad_added_to_main_grad", False):
                effective_grad = parameter.main_grad
            else:
                effective_grad = (
                    parameter.grad
                    if parameter.grad is not None
                    else parameter.main_grad
                )
            grads.append(effective_grad.detach().clone())
            # Match Megatron's optimizer interface, which consumes main_grad.
            parameter.grad = effective_grad
        torch.optim.SGD(attention.parameters(), lr=0.01).step()
        updated = [parameter.detach().clone() for parameter in attention.parameters()]
        return (
            mixed_qkv.detach(),
            hidden_states.grad.detach(),
            grads,
            updated,
            loss.detach(),
            forward_count[0],
            handle,
        )

    no_recompute = run("no_recompute")
    no_recompute_repeat = run("no_recompute")
    on_demand = run("on_demand")
    proactive = run("proactive")

    # Establish the deterministic reference baseline before comparing modes.
    for left, right in zip(no_recompute[:5], no_recompute_repeat[:5]):
        if isinstance(left, list):
            for left_value, right_value in zip(left, right):
                if left_value is None:
                    assert right_value is None
                else:
                    torch.testing.assert_close(left_value, right_value, rtol=0, atol=0)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    for result in (on_demand, proactive):
        for index in (0, 1, 4):
            torch.testing.assert_close(result[index], no_recompute[index], rtol=0, atol=0)
        for actual, expected in zip(result[2], no_recompute[2]):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(result[3], no_recompute[3]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert no_recompute[5] == 1
    assert on_demand[5] == 2
    assert proactive[5] == 2  # initial forward plus one proactive replay
    assert proactive[6].state == "consumed"
    event_names = [event["event"] for event in proactive[6].events]
    assert event_names.count("cache_hit") == 1
    assert event_names.count("cache_consumed") == 1
    assert proactive[6]._cached_outputs is None


def test_proactive_recompute_microbatch_identity_order_and_bounded_cache():
    module = torch.nn.Linear(4, 4, bias=False).cuda()
    set_proactive_recompute_cache_limit(1)
    try:
        x0 = torch.randn(3, 4, device="cuda", requires_grad=True)
        x1 = torch.randn(3, 4, device="cuda", requires_grad=True)
        with proactive_recompute_context(step=12, microbatch=0):
            handle0 = ProactiveCheckpoint(module)
            output0 = handle0(x0)
        with proactive_recompute_context(step=12, microbatch=1):
            handle1 = ProactiveCheckpoint(module)
            output1 = handle1(x1)

        assert handle0.metadata == {"step": 12, "microbatch": 0}
        assert handle1.metadata == {"step": 12, "microbatch": 1}
        assert handle0.checkpoint_id != handle1.checkpoint_id

        # Trigger A: immediately after the checkpoint's original forward.
        assert prefetch_recompute(handle0.checkpoint_id, trigger_site="after_forward")
        assert proactive_recompute_cache_stats()["cached_checkpoint_count"] == 1
        # A second graph is refused while the configured one-entry cache is occupied.
        assert not prefetch_recompute(handle1.checkpoint_id, trigger_site="cache_limit_probe")
        assert handle1.state == "forward_complete"

        output0.sum().backward()
        events0 = handle0.events
        complete0 = next(event for event in events0 if event["event"] == "prefetch_complete")
        consume0 = next(event for event in events0 if event["event"] == "cache_consumed")
        assert complete0["timestamp"] < consume0["timestamp"]
        assert handle0.state == "consumed"
        assert proactive_recompute_cache_stats()["cached_checkpoint_count"] == 0

        # Trigger B: an external caller schedules replay while another microbatch
        # is doing independent work. All operations use the current CUDA stream.
        _ = torch.sin(x1).sum()
        assert prefetch_recompute(
            handle1.checkpoint_id, trigger_site="other_microbatch_compute"
        )
        output1.sum().backward()
        assert handle1.state == "consumed"
        assert proactive_recompute_cache_stats()["saved_tensor_bytes"] == 0
        assert_no_live_proactive_recompute_checkpoints()

        # A new training step does not see the previous step's handle or graph.
        with proactive_recompute_context(step=13, microbatch=0):
            handle2 = ProactiveCheckpoint(module)
            output2 = handle2(torch.randn(3, 4, device="cuda", requires_grad=True))
        output2.sum().backward()  # omitted prefetch uses ordinary on-demand replay
        assert any(event["event"] == "fallback" for event in handle2.events)
        assert_no_live_proactive_recompute_checkpoints()
    finally:
        set_proactive_recompute_cache_limit(None)


def test_proactive_checkpoint_backward_exception_rolls_back_parameter_grads():
    class FailBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, grad_output):
            raise RuntimeError("injected backward error")

    module = torch.nn.Linear(4, 3).cuda()

    def run(value):
        return FailBackward.apply(module(value))

    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    handle = ProactiveCheckpoint(run, parameters=module.parameters())
    output = handle(x)
    assert handle.prefetch()
    with pytest.raises(RuntimeError, match="injected backward error"):
        output.sum().backward()
    assert module.weight.grad is None
    assert handle.state == "backward_failed"
    assert handle._cached_outputs is None


def test_proactive_checkpoint_backward_exception_with_main_grad_is_fatal():
    class FailBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, grad_output):
            raise RuntimeError("injected fused backward error")

    module = torch.nn.Linear(4, 3).cuda()
    for parameter in module.parameters():
        parameter.main_grad = torch.zeros_like(parameter)

    def run(value):
        return FailBackward.apply(module(value))

    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    handle = ProactiveCheckpoint(run, parameters=module.parameters())
    output = handle(x)
    assert handle.prefetch()
    with pytest.raises(RuntimeError, match="terminate this training attempt"):
        output.sum().backward()
    assert handle.state == "backward_failed"
    failure = next(event for event in handle.events if event["event"] == "backward_failed")
    assert failure["main_grad_state_may_be_partial"] is True
    assert handle._cached_outputs is None


def test_proactive_checkpoint_explicit_release_cleans_unconsumed_cache():
    x = torch.randn(3, device="cuda", requires_grad=True)
    handle = ProactiveCheckpoint(lambda value: value.sin())
    output = handle(x)
    assert handle.prefetch()
    handle.release()
    assert handle.state == "released"
    assert handle._cached_outputs is None
    with pytest.raises(RuntimeError, match="released"):
        output.sum().backward()


def test_proactive_checkpoint_shared_parameter_and_aliased_inputs():
    weight = torch.nn.Parameter(torch.randn(4, 4, device="cuda"))
    x = torch.randn(2, 4, device="cuda", requires_grad=True)

    def run(first, second):
        return torch.nn.functional.linear(first, weight) + torch.nn.functional.linear(
            second, weight
        )

    handle = ProactiveCheckpoint(run, parameters=[weight, weight])
    output = handle(x, x)
    assert handle.prefetch()
    output.sum().backward()
    expected_input_grad = 2 * weight.detach().sum(dim=0).expand_as(x)
    torch.testing.assert_close(x.grad, expected_input_grad)
    assert weight.grad is not None
    assert len(handle.parameters) == 1


def test_proactive_checkpoint_repeated_steps_release_all_invocations():
    module = torch.nn.Linear(6, 4).cuda()
    handles = []
    for _ in range(5):
        module.zero_grad(set_to_none=True)
        x = torch.randn(3, 6, device="cuda", requires_grad=True)
        handle = ProactiveCheckpoint(module, parameters=module.parameters())
        handles.append(handle)
        output = handle(x)
        assert handle.prefetch()
        output.square().sum().backward()
        assert handle.state == "consumed"
        assert handle._cached_outputs is None
        assert handle._cached_inputs is None
        assert handle.checkpoint_id not in pending_recompute_checkpoints()
    assert len({handle.checkpoint_id for handle in handles}) == len(handles)
