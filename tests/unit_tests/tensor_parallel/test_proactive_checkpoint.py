"""Single-device correctness tests for the experimental proactive checkpoint API."""

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    ProactiveCheckpoint,
    checkpoint,
    pending_recompute_checkpoints,
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
