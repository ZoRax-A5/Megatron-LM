# 重计算与通信重叠：使用报告

## 全局前提

所有自动训练路径都需要：

```text
--recompute-comm-overlap
--recompute-granularity selective
--recompute-modules layernorm moe_act core_attn
```

总开关不会自动启用重计算。选择模块应以显存峰值和双卡/目标集群 profiling 为准；不要重算会再次发起同一通信域 collective 的完整模块。

## EP All-to-All

```text
--overlap-moe-expert-parallel-comm
--recompute-comm-overlap
--recompute-granularity selective
--recompute-modules layernorm moe_act core_attn
--moe-grouped-gemm
```

要求 EP size 大于 1，token dispatcher 为 alltoall 或 flex，精度为 bf16/fp16。不能使用 full-layer/whole-MoE 重计算；`moe_act` 需要 grouped GEMM。

## PP P2P

```text
--pipeline-model-parallel-size 2
--num-layers-per-virtual-pipeline-stage <N>
--recompute-comm-overlap
--pp-recompute-comm-overlap
--recompute-granularity selective
--recompute-modules layernorm core_attn
```

当前仅支持 interleaved pipeline，且不能启用 batch P2P。训练参数转换会为该模式设置 `batch_p2p_comm=False`。

## DDP 梯度 All-Reduce / Reduce-Scatter

普通 DDP All-Reduce：

```text
--recompute-comm-overlap
--recompute-grad-sync-overlap
--recompute-granularity selective
--recompute-modules layernorm core_attn
```

分布式优化器 Reduce-Scatter 再增加：

```text
--use-distributed-optimizer
```

专用开关会自动启用 `overlap_grad_reduce`，无需重复传 `--overlap-grad-reduce`。

## Megatron-FSDP 参数 All-Gather

```text
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--recompute-comm-overlap
--megatron-fsdp-recompute-ag-overlap
--recompute-granularity selective
--recompute-modules layernorm core_attn
```

不要同时使用 `--fsdp-double-buffer`；MXFP8 parameter All-Gather 尚不支持。重计算预取深度固定为相邻一个 backward-order 桶组，以保护显存峰值。

## TP All-Gather / Reduce-Scatter API

TP 路径需要 sequence parallel，且当前只支持原生 MCore Linear，不支持 Transformer Engine Userbuffers。调用方要提供与待通信张量无依赖、且内部不启动 TP collective 的重计算函数：

```python
from megatron.core.tensor_parallel import (
    TensorParallelRecomputeCommOverlapConfig,
    tensor_parallel_recompute_comm_overlap,
)

overlap = TensorParallelRecomputeCommOverlapConfig(
    recompute_comm_overlap=True,
    tensor_parallel_overlap=True,
    sequence_parallel=True,
)

with tensor_parallel_recompute_comm_overlap(overlap, recompute_independent_activation):
    output = native_sequence_parallel_linear(...)
```

上下文只消费一次回调：第一个被允许的原生 All-Gather 或 Reduce-Scatter 获得重计算窗口。若需要多个窗口，应为每个独立回调建立单独上下文。

也可以直接使用两阶段原语：

```python
from megatron.core.tensor_parallel import (
    launch_all_gather_from_sequence_parallel_region,
    overlap_with_pending_tensor_parallel_collective,
)

pending = launch_all_gather_from_sequence_parallel_region(local_input, group=tp_group)
gathered, recomputed = overlap_with_pending_tensor_parallel_collective(
    pending, recompute_independent_activation
)
```

## 动态专家显存均衡建议

```text
--moe-router-enable-expert-bias
--moe-router-bias-update-rate 1e-3
--moe-token-dispatcher-type flex
--moe-flex-dispatcher-backend hybridep
--use-transformer-engine-op-fuser
--moe-expert-rank-capacity-factor <实测值>
```

容量因子应由 routed-token 分布和各 rank 峰值显存共同确定，并为通信重叠期间共存的 attention activation 留出余量。先在较保守值上验证无频繁回退，再逐步收紧。
