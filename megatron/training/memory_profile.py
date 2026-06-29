# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Event-based, rank-local CUDA memory profiling for training diagnostics."""

from __future__ import annotations

import functools
import json
import logging
import pickle
import socket
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


StorageKey = tuple[int, int]


def _storage_key(tensor: torch.Tensor) -> tuple[StorageKey, int] | None:
    """Return a stable key and size for a CUDA tensor's underlying storage."""
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        return None
    storage = tensor.untyped_storage()
    size = storage.nbytes()
    if size == 0:
        return None
    return (storage.data_ptr(), size), size


def _walk_tensors(value: Any, seen: set[int] | None = None) -> Iterable[torch.Tensor]:
    """Yield tensors from common optimizer containers without traversing arbitrary objects."""
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_tensors(key, seen)
            yield from _walk_tensors(item, seen)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _walk_tensors(item, seen)


@dataclass
class _PackedTensor:
    tensor: torch.Tensor
    token: int


class _SavedTensorTracker:
    """Track storages retained by autograd for backward."""

    def __init__(self) -> None:
        self._next_token = 0
        self._active: dict[int, tuple[StorageKey, int] | None] = {}

    def pack(self, tensor: torch.Tensor) -> _PackedTensor:
        token = self._next_token
        self._next_token += 1
        self._active[token] = _storage_key(tensor)
        return _PackedTensor(tensor=tensor, token=token)

    def unpack(self, packed: _PackedTensor) -> torch.Tensor:
        self._active.pop(packed.token, None)
        return packed.tensor

    def storages(self) -> dict[StorageKey, int]:
        return {key: size for entry in self._active.values() if entry for key, size in [entry]}


class TrainingMemoryProfiler:
    """Collect CUDA allocator samples for a bounded training-iteration window.

    Samples are written independently by every rank. Deep mode synchronizes CUDA at each
    sample, tracks tensors saved for backward, inventories persistent model/optimizer tensors,
    and saves a PyTorch allocator snapshot when the profile window finishes.
    """

    def __init__(
        self,
        *,
        mode: str,
        warmup_iters: int,
        profile_iters: int,
        output_dir: str,
        start_iteration: int,
        model: Iterable[torch.nn.Module],
        optimizer: Any,
        rank_metadata: Mapping[str, Any],
        history_max_entries: int = 100000,
    ) -> None:
        if mode not in ("light", "deep"):
            raise ValueError(f"Unsupported memory profile mode: {mode}")
        if warmup_iters < 0 or profile_iters <= 0:
            raise ValueError(
                "Memory profile warmup must be non-negative and profile iters positive"
            )

        self.mode = mode
        self.deep = mode == "deep"
        self.profile_start_iteration = start_iteration + warmup_iters
        self.profile_end_iteration = self.profile_start_iteration + profile_iters
        self.output_dir = Path(output_dir)
        self.model = list(model)
        self.optimizer = optimizer
        self.metadata = dict(rank_metadata)
        self.rank = int(self.metadata["rank"])
        self.metadata.setdefault("hostname", socket.gethostname())
        self.history_max_entries = history_max_entries
        self._prepare_output_dir()

        self.active = False
        self.finished = False
        self.current_iteration: int | None = None
        self._origin_ns: int | None = None
        self._event_index = 0
        self._history_enabled = False
        self._saved_tensors = _SavedTensorTracker()
        self._optimizer_original_methods: dict[str, Any] = {}
        self._backward_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._file = open(self.output_dir / f"rank{self.rank}.jsonl", "w", encoding="utf-8")

        self._instrument_optimizer()
        self._register_moe_backward_hooks()

    def _prepare_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            stale_paths = [
                *self.output_dir.glob("rank*.jsonl"),
                *self.output_dir.glob("rank*_snapshot.pickle"),
                self.output_dir / "all_samples.csv",
            ]
            for path in stale_paths:
                if path.exists():
                    path.unlink()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def begin_iteration(self, iteration: int) -> None:
        """Activate profiling when the configured warmup window has completed."""
        self.current_iteration = iteration
        if self.finished or iteration < self.profile_start_iteration:
            return
        if iteration >= self.profile_end_iteration:
            self._finish()
            return
        if not self.active:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if self.deep:
                torch.cuda.synchronize()
            self._origin_ns = time.perf_counter_ns()
            torch.cuda.reset_peak_memory_stats()
            self.active = True
            self._start_history()
        self.sample("iteration_start")

    def end_iteration(self, iteration: int) -> None:
        """Flush an iteration and finish after the configured number of samples."""
        if not self.active or self.current_iteration != iteration:
            return
        self.sample("iteration_end")
        self._file.flush()
        if iteration + 1 >= self.profile_end_iteration:
            self._finish()

    def saved_tensors_context(self):
        """Return an autograd saved-tensor context when deep profiling is active."""
        if not self.active or not self.deep:
            return nullcontext()
        return torch.autograd.graph.saved_tensors_hooks(
            self._saved_tensors.pack, self._saved_tensors.unpack
        )

    def sample(
        self,
        event: str,
        *,
        microbatch: int | None = None,
        vp_stage: int | None = None,
        layer: int | None = None,
        communication_tensors: Any = None,
    ) -> None:
        """Record one event at the current rank and iteration."""
        if not self.active or self.finished:
            return
        if self.deep:
            torch.cuda.synchronize()

        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        peak_allocated = int(torch.cuda.max_memory_allocated())
        device_used = self._device_memory_used()
        now_ns = time.perf_counter_ns()
        record: dict[str, Any] = {
            **self.metadata,
            "mode": self.mode,
            "iteration": self.current_iteration,
            "event_index": self._event_index,
            "event": event,
            "microbatch": microbatch,
            "vp_stage": vp_stage,
            "layer": layer,
            "elapsed_ms": (now_ns - self._origin_ns) / 1.0e6,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "interval_peak_allocated_bytes": peak_allocated,
            "device_used_bytes": device_used,
        }
        if self.deep:
            record["breakdown_bytes"] = self._breakdown(
                allocated=allocated,
                reserved=reserved,
                device_used=device_used,
                communication_tensors=communication_tensors,
            )
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._event_index += 1
        torch.cuda.reset_peak_memory_stats()

    def close(self) -> None:
        """Finalize output and restore wrapped optimizer methods."""
        if not self.finished:
            self._finish()

    def _persistent_tensors(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        parameters = list(
            _walk_tensors([list(model_chunk.parameters()) for model_chunk in self.model])
        )
        optimizer_tensors: list[torch.Tensor] = []
        optimizers = getattr(self.optimizer, "chained_optimizers", [self.optimizer])
        for optimizer in optimizers:
            optimizer_tensors.extend(_walk_tensors(getattr(optimizer, "param_groups", [])))
            optimizer_tensors.extend(_walk_tensors(getattr(optimizer, "state", {})))

        gradients: list[torch.Tensor] = []
        for parameter in [*parameters, *optimizer_tensors]:
            grad = getattr(parameter, "grad", None)
            main_grad = getattr(parameter, "main_grad", None)
            if isinstance(grad, torch.Tensor):
                gradients.append(grad)
            if isinstance(main_grad, torch.Tensor):
                gradients.append(main_grad)
        return parameters, gradients, optimizer_tensors

    @staticmethod
    def _claim_tensors(tensors: Iterable[torch.Tensor], claimed: set[StorageKey]) -> int:
        total = 0
        for tensor in tensors:
            entry = _storage_key(tensor)
            if entry is None:
                continue
            key, size = entry
            if key not in claimed:
                claimed.add(key)
                total += size
        return total

    @staticmethod
    def _claim_storages(storages: Mapping[StorageKey, int], claimed: set[StorageKey]) -> int:
        total = 0
        for key, size in storages.items():
            if key not in claimed:
                claimed.add(key)
                total += size
        return total

    def _breakdown(
        self, *, allocated: int, reserved: int, device_used: int, communication_tensors: Any
    ) -> dict[str, int]:
        parameters, gradients, optimizer_tensors = self._persistent_tensors()
        claimed: set[StorageKey] = set()
        categories = {
            "parameter": self._claim_tensors(parameters, claimed),
            "gradient": self._claim_tensors(gradients, claimed),
            "optimizer": self._claim_tensors(optimizer_tensors, claimed),
        }

        communication = list(_walk_tensors(communication_tensors))
        categories["communication"] = self._claim_tensors(communication, claimed)
        saved = self._saved_tensors.storages()
        categories["saved_activation"] = self._claim_storages(saved, claimed)
        known_allocated = sum(categories.values())
        categories["other_torch"] = max(allocated - known_allocated, 0)
        categories["allocator_cache"] = max(reserved - allocated, 0)
        categories["external_cuda"] = max(device_used - reserved, 0)
        categories["unaccounted_overlap"] = max(known_allocated - allocated, 0)
        return categories

    @staticmethod
    def _device_memory_used() -> int:
        if hasattr(torch.cuda, "device_memory_used"):
            return int(torch.cuda.device_memory_used())
        free, total = torch.cuda.mem_get_info()
        return int(total - free)

    def _instrument_optimizer(self) -> None:
        if self.optimizer is None:
            return
        for method_name in (
            "prepare_grads",
            "get_grad_norm",
            "_compute_grad_norms_by_group",
            "count_zeros",
            "step_with_ready_grads",
        ):
            original = getattr(self.optimizer, method_name, None)
            if not callable(original):
                continue
            self._optimizer_original_methods[method_name] = original

            @functools.wraps(original)
            def wrapped(*args, _method=original, _name=method_name, **kwargs):
                self.sample(f"optimizer_{_name}_start")
                result = _method(*args, **kwargs)
                self.sample(f"optimizer_{_name}_end")
                return result

            setattr(self.optimizer, method_name, wrapped)

    def _register_moe_backward_hooks(self) -> None:
        for model_chunk in self.model:
            modules = model_chunk.modules() if hasattr(model_chunk, "modules") else ()
            for module in modules:
                if not hasattr(module, "token_dispatcher") or not hasattr(module, "experts"):
                    continue

                def backward_start(current_module, grad_output):
                    self.sample(
                        "moe_backward_start",
                        layer=getattr(current_module, "layer_number", None),
                        communication_tensors=grad_output,
                    )

                def backward_end(current_module, grad_input, grad_output):
                    self.sample(
                        "moe_backward_end",
                        layer=getattr(current_module, "layer_number", None),
                        communication_tensors=[grad_input, grad_output],
                    )

                self._backward_hooks.append(module.register_full_backward_pre_hook(backward_start))
                self._backward_hooks.append(module.register_full_backward_hook(backward_end))

    def _start_history(self) -> None:
        if not self.deep:
            return
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=self.history_max_entries,
                clear_history=True,
            )
            self._history_enabled = True
        except (AttributeError, RuntimeError, TypeError) as error:
            logger.warning("Unable to start CUDA memory history on rank %s: %s", self.rank, error)

    def _finish(self) -> None:
        if self.finished:
            return
        if self._history_enabled:
            try:
                snapshot = torch.cuda.memory._snapshot()
                with open(self.output_dir / f"rank{self.rank}_snapshot.pickle", "wb") as file:
                    pickle.dump(snapshot, file)
                torch.cuda.memory._record_memory_history(enabled=None)
            except (AttributeError, OSError, RuntimeError, TypeError) as error:
                logger.warning(
                    "Unable to save CUDA memory snapshot on rank %s: %s", self.rank, error
                )
        for method_name, original in self._optimizer_original_methods.items():
            setattr(self.optimizer, method_name, original)
        self._optimizer_original_methods.clear()
        for hook in self._backward_hooks:
            hook.remove()
        self._backward_hooks.clear()
        self.active = False
        self.finished = True
        self._file.flush()
        self._file.close()
