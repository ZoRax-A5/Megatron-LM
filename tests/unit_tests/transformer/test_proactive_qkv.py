"""Single-GPU correctness test for the local Megatron QKV checkpoint boundary."""

import json
import os
from pathlib import Path

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import (
    get_recompute_checkpoint,
    pending_recompute_checkpoints,
    prefetch_recompute,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_local_qkv_no_recompute_on_demand_and_proactive_match(tmp_path):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    try:
        layer_spec = get_gpt_layer_local_spec()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            use_cpu_initialization=True,
            add_bias_linear=False,
            params_dtype=torch.float32,
            transformer_impl="local",
            recompute_granularity="selective",
            recompute_modules=["attn_qkv"],
        )
        attention = (
            SelfAttention(
                config, layer_spec.submodules.self_attention.submodules, layer_number=1
            )
            .cuda()
            .train()
        )
        config.recompute_granularity = None
        with torch.no_grad():
            warmup_qkv, _ = attention.get_query_key_value_tensors(
                torch.randn(5, 2, 16, device="cuda"), split_qkv=False
            )
        del warmup_qkv
        torch.cuda.synchronize()
        config.recompute_granularity = "selective"
        initial_state = {
            name: None if value is None else value.detach().clone()
            for name, value in attention.state_dict().items()
        }
        base_input = torch.randn(5, 2, 16, device="cuda")
        call_count = 0

        def count_linear_calls(module, args, output):
            nonlocal call_count
            call_count += 1

        hook = attention.linear_qkv.register_forward_hook(count_linear_calls)

        def run_variant(mode):
            nonlocal call_count
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            memory = {
                "allocated_before": torch.cuda.memory_allocated(),
                "reserved_before": torch.cuda.memory_reserved(),
            }
            attention.load_state_dict(initial_state)
            for parameter in attention.parameters():
                parameter.grad = None
            config.recompute_granularity = None if mode == "none" else "selective"
            config.proactive_recompute = mode == "proactive"
            call_count = 0
            hidden = base_input.detach().clone().requires_grad_(True)
            mixed_qkv, _ = attention.get_query_key_value_tensors(
                hidden, split_qkv=False
            )
            output = mixed_qkv.detach().cpu()
            calls_after_forward = call_count
            calls_after_prefetch = call_count
            checkpoint_id = None
            handle = None
            if mode == "proactive":
                pending = pending_recompute_checkpoints()
                assert len(pending) == 1
                checkpoint_id = pending[0]
                handle = get_recompute_checkpoint(checkpoint_id)
                assert prefetch_recompute(checkpoint_id)
                calls_after_prefetch = call_count
                assert call_count == 2
                memory["allocated_after_prefetch"] = torch.cuda.memory_allocated()
                memory["reserved_after_prefetch"] = torch.cuda.memory_reserved()
            mixed_qkv.square().sum().backward()
            calls_after_backward = call_count
            grads = [
                None if parameter.grad is None else parameter.grad.detach().cpu()
                for parameter in attention.parameters()
            ]
            input_grad = hidden.grad.detach().cpu()
            optimizer = torch.optim.SGD(attention.parameters(), lr=0.01)
            optimizer.step()
            torch.cuda.synchronize()
            memory.update(
                allocated_after_backward=torch.cuda.memory_allocated(),
                reserved_after_backward=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
            )
            updated = [parameter.detach().cpu() for parameter in attention.parameters()]
            return {
                "output": output,
                "input_grad": input_grad,
                "grads": grads,
                "updated": updated,
                "forward_calls": calls_after_forward,
                "backward_calls": calls_after_backward,
                "prefetch_calls": calls_after_prefetch,
                "checkpoint_id": checkpoint_id,
                "handle": handle,
                "memory": memory,
            }

        no_recompute = run_variant("none")
        no_recompute_repeat = run_variant("none")
        on_demand = run_variant("on_demand")
        proactive = run_variant("proactive")
        hook.remove()

        for candidate in (no_recompute_repeat, on_demand, proactive):
            torch.testing.assert_close(candidate["output"], no_recompute["output"])
            torch.testing.assert_close(
                candidate["input_grad"], no_recompute["input_grad"]
            )
            for expected, actual in zip(no_recompute["grads"], candidate["grads"]):
                if expected is None:
                    assert actual is None
                else:
                    torch.testing.assert_close(actual, expected)
            for expected, actual in zip(no_recompute["updated"], candidate["updated"]):
                torch.testing.assert_close(actual, expected)

        assert on_demand["forward_calls"] == 1
        assert on_demand["backward_calls"] == 2
        assert proactive["forward_calls"] == 1
        assert proactive["backward_calls"] == 2
        events = [event["event"] for event in proactive["handle"].events]
        assert events.count("prefetch_complete") == 1
        assert events.count("cache_hit") == 1
        assert events.count("cache_consumed") == 1
        assert proactive["checkpoint_id"] not in pending_recompute_checkpoints()
        result_dir = os.getenv("MEGATRON_PROACTIVE_RESULT_DIR")
        if result_dir:
            output_dir = Path(result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = {}
            for name, result in (
                ("no_recompute", no_recompute),
                ("no_recompute_repeat", no_recompute_repeat),
                ("on_demand", on_demand),
                ("proactive", proactive),
            ):
                summary[name] = {
                    "original_forward_calls": result["forward_calls"],
                    "prefetch_replay_calls": result["prefetch_calls"]
                    - result["forward_calls"],
                    "backward_replay_calls": result["backward_calls"]
                    - result["prefetch_calls"],
                    "memory_bytes": result["memory"],
                }
            (output_dir / "qkv_metrics.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            proactive["handle"].write_events(output_dir / "qkv_proactive_events.json")
    finally:
        Utils.destroy_model_parallel()
