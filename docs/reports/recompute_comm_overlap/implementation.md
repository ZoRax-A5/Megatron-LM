# 重计算与通信重叠：实现过程报告

## 目标与设计约束

目标是在 MoE 激活重计算降低显存占用的前提下，把额外重计算放入硬件争用较低、且依赖关系允许的通信窗口。所有功能默认关闭；全局开关只授权重排，不会隐式启用激活重计算。

实现遵循三个约束：

1. 只跨独立微批次或独立张量依赖移动操作，不能跨越真实数据依赖。
2. 异步通信缓冲区在 wait 前保持存活，异常路径也必须清空通信句柄。
3. 重叠不能抵消显存收益；FSDP 重计算预取被限制为相邻一个反向桶组。

## 分阶段实现

### 阶段 1：EP All-to-All

在 combined-1F1B MoE 调度中，开启总开关后先启动当前前向微批次的 dispatch All-to-All，再执行上一微批次的专家反向。专家反向触发的 `moe_act` 或 LayerNorm 选择性重计算可处于 A2A 飞行窗口内。combine A2A 原本已先于注意力反向启动，因此可覆盖注意力侧选择性重计算。

同一微批次内的 backward A2A 与生成其输入的反向计算存在硬依赖，本实现不声称将它与激活重计算重叠。已有 delay-wgrad 路径隐藏的是权重梯度计算，也不等价于本功能。

关键文件：

- `megatron/core/models/common/model_chunk_schedule_plan.py`
- `megatron/core/transformer/transformer_config.py`

### 阶段 2：PP P2P

在 interleaved pipeline 调度中，将 PP 专用开关并入已有异步 P2P 请求协议。前向 send/recv 启动后保持句柄未完成，随后进入反向及其激活重计算。非 interleaved 调度和 batch P2P 明确拒绝该模式。

训练参数转换同时保证 PP 专用开关会关闭 `batch_p2p_comm`。由于训练入口默认启用 pipeline output deallocation，send 端在部分路径上必须提前等待以保证生命周期；下一前向微批次的异步 receive 仍可覆盖反向重计算，因此不把它描述成所有双向 P2P 都被完整隐藏。

关键文件：

- `megatron/core/pipeline_parallel/p2p_communication.py`
- `megatron/core/pipeline_parallel/schedules.py`
- `megatron/training/argument_utils.py`
- `megatron/training/yaml_arguments.py`

### 阶段 3：梯度 All-Reduce / Reduce-Scatter

新增显式语义开关并复用成熟的 DDP 异步桶协议。一个梯度桶就绪后启动 All-Reduce；启用分布式优化器时启动 Reduce-Scatter。通信与更早层的 backward 并行，其中自然包括这些层触发的激活重计算。

该路径不改变梯度数学语义，也保留首批次桶就绪顺序学习行为。

关键文件：

- `megatron/core/distributed/distributed_data_parallel_config.py`
- `megatron/core/distributed/distributed_data_parallel.py`
- `megatron/training/arguments.py`

### 阶段 4：Megatron-FSDP 参数 All-Gather

FSDP 检测 `TrainingState.PRE_BACKWARD`，即反向期间的激活重放。此时原有 forward-order 预取仍被关闭，因为重计算按反向消费顺序发生；新路径改为预取下一个 backward-order 参数桶组。预取阈值固定为 1，从而只多物化相邻桶组。

仅支持 `optim_grads_params` 全参数分片；禁止 FSDP double buffer 和 MXFP8 参数 All-Gather 组合。

关键文件：

- `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`
- `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`

### 阶段 5：TP All-Gather / Reduce-Scatter

实现了可等待一次的两阶段原语：launch 返回持有独立输入/输出缓冲区的 pending handle，wait 通过自定义 autograd Function 补上逆通信。原生 sequence-parallel Linear 和 RowParallelLinear 已识别活动重计算上下文，在 launch 与 wait 之间消费一次独立重计算回调。

为防死锁，回调不得启动 TP collective；嵌套上下文被拒绝；非 CUDA/NCCL 后端退化为先 wait 再重计算；异常时仍会 drain pending handle。Transformer Engine Userbuffers 暂不支持。

这是完整的底层原语和原生 MCore Linear 接入，但训练调度器不会自动猜测哪个激活可独立重算。调用方必须显式提供回调和上下文。

关键文件：

- `megatron/core/tensor_parallel/mappings.py`
- `megatron/core/tensor_parallel/recompute_comm_overlap.py`
- `megatron/core/tensor_parallel/layers.py`

## 能力边界

| 通信域 | 实现状态 | 自动训练路径 |
| --- | --- | --- |
| EP dispatch/combine All-to-All | 已集成 | combined-1F1B |
| PP P2P | 已集成 | interleaved pipeline |
| DP gradient All-Reduce | 已集成 | DDP async buckets |
| DP gradient Reduce-Scatter | 已集成 | distributed optimizer buckets |
| FSDP parameter All-Gather | 已集成 | Megatron-FSDP recompute hook |
| TP sequence All-Gather/Reduce-Scatter | 原语与 Linear 接入完成 | 需要显式重计算上下文 |
| backward A2A × 同微批次重计算 | 未实现 | 存在硬数据依赖 |

## 动态专家显存不均衡

生产基线采用两层控制：`--moe-router-enable-expert-bias` 纠正长期 token 偏斜，`--moe-expert-rank-capacity-factor` 给每个 EP rank 的临时 token 缓冲设置硬上界。后者需要 flex dispatcher、HybridEP 和 Transformer Engine op fuser；超预算步骤由 paged-stash 回退重跑，避免静默丢 token。

后续可实现带滞回的内存反馈控制器：维护每个专家 rank 的每 token 字节数和空闲显存 EMA，只把慢变化的拥塞项加入 expert bias。该方案需要单独做收敛与专家利用率评估，因此未混入本次安全基线。
