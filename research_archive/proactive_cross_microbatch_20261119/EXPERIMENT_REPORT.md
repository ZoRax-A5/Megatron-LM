# 跨 micro-batch 主动重计算：单 GPU 原型

## 1. 核心摘要（Executive Summary）

**P8 核心机制 PASS，工程集成与扩展范围尚未验证。** 在单 GPU 的真实 BF16 Transformer Engine 融合 QKV 模块上，B 的 Forward 内部触发了 A 的 Replay；A 的 Backward 消费提前计算图。原始顺序、调整顺序 On-demand 与调整顺序 Proactive 的 loss、完整输入梯度、模块全部参数有效梯度和 SGD 更新后参数逐元素一致。

- A 的 QKV 原始 Forward 1 次、主动 Replay 1 次、Backward Replay 0 次；缓存命中、消费、释放各 1 次。
- CUDA Event 表明 Replay 的 GPU stream 标记位于 B Forward 区间内，且早于 A Backward；同一 stream 的顺序执行不代表 kernel 并发。
- 改变调度与主动 Replay 分别增加峰值 allocated 56,320 和 66,048 bytes；这是单次小模块实验的进程内测量。
- 建议审阅原型后，再验证真实训练 schedule、多个 step 和分布式行为；当前没有修改默认训练路径。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 设备和精度 | 单 GPU，A100 80GB，BF16 |
| 模块 | Megatron `SelfAttention` 的真实 TE 融合 LayerNorm/ColumnParallelLinear QKV；hidden 64，4 heads，输入 `[64,2,64]` |
| 比较 | 原始 F(A) B(A) F(B) B(B)；调整 F(A) F(B) B(A) B(B) 按需重放；调整顺序下 B QKV 前置 hook 触发 A Replay |
| 梯度 | 两个 loss 各除以 2，累积参数有效梯度；SGD 一次更新 |
| 时间与显存 | 当前 CUDA stream 上插入 CUDA Event，CPU 用单调时钟，NVTX 标注范围；CUDA allocator peak allocated/reserved |

原型是独立、显式运行的测试调度，默认训练路径未启用。每组使用相同初始参数和输入，原始基线重复一次；计数按实际 QKV 调用阶段记录。Event 时间是 stream 上标记的执行顺序，CPU 时间是调用和提交时刻，二者含义不同。

## 3. 正确性与观测完整性验证

| 检查 | 结果 |
|---|---|
| 基线重复性 | 两次原始调度 loss、输入梯度、全部有效参数梯度、更新后参数逐位相同 |
| 三组数值 | 两个 loss、两个输入完整梯度、全部模块有效参数梯度和更新参数均 `rtol=0, atol=0` 一致 |
| A 执行次数 | original 1、proactive replay 1、backward replay 0；命中、消费、释放各 1 |
| 身份 | 目标 checkpoint 的真实上下文为 step 7、microbatch 0、layer 1、attn_qkv、rank 0；触发在 B 的 QKV forward pre-hook |
| 生命周期 | A consumed 后缓存清空；B 未预取，按需回退；两者反向后注册表无活动句柄 |

测试使用 TE fused weight gradient accumulation 所需 `main_grad`，并在两次反向中累积。随机种子相同且本模块测试没有随机层；本实验没有单独覆盖随机算子的跨 micro-batch RNG 行为。

## 4. 性能评估与定量测量

| 配置 | peak allocated bytes（单次） | 相对原始 | peak reserved bytes（单次） | 相对原始 |
|---|---:|---:|---:|---:|
| 原始调度 On-demand | 4,965,376 | 基线 | 23,068,672 | 基线 |
| 调整调度 On-demand | 5,021,696 | +1.13% | 23,068,672 | 0% |
| 调整调度 Proactive | 5,087,744 | +2.46% | 23,068,672 | 0% |

- A 输出缓存为 49,152 bytes；引用的输出及 Saved Tensor backing storage 去重估计为 107,520 bytes，该值包括已有共享存储，并非新增显存。
- Prefetch 前后 allocated 由 4,529,152 增至 4,595,200 bytes，增量 66,048 bytes；reserved 无变化。缓存 CPU 侧生命周期为 2.257 ms。
- 这是一个小模块、每组一次的诊断测量，没有独立重复样本或吞吐结论；allocator 状态及测量粒度限制了外推。

## 5. 深度分析与机制探讨

以 A Forward 起始 CUDA Event 为零点，B Forward start 为 0.526 ms、A Replay start/end 为 0.689/1.777 ms、B Forward end 为 2.074 ms、A Backward start 为 2.089 ms。**直接实测的 stream 顺序**满足 B 开始 < Replay 开始 < Replay 结束 < B 结束 < A Backward。CPU 调用时间也满足相同顺序。

B 的前置 hook 在真实 TE QKV 调用前执行；此时 A 的 checkpoint 已登记并通过 step 和 micro-batch 身份确认。Replay 保留可微图，A Backward 对该图求导。B 的计算随后继续，B 自身在 Backward 时使用按需路径。

保留两个 Forward 的图会使峰值比逐个 Forward/Backward 增加；提前图及其 Saved Tensors 再增加驻留。这是与观测一致的机制解释，当前单次 allocator 统计不足以细分所有内存来源。

## 6. 局限性、决策建议与后续工作

本实验只覆盖单层 QKV、小张量、单 GPU、两个 micro-batch 和一次优化器更新；它不是完整 Megatron 训练 schedule，也未覆盖跨 step、多 GPU 或异常后的分布式恢复。未测试 Dropout、TE 其他配置、独立 stream 和真实 kernel 并发。Event 在同一 stream 上记录边界，并非 profiler kernel trace。

建议下一步先在多个 step、更多输入与随机算子条件下验证身份与 RNG，再将可控调度接入真实训练并逐步检查 TP collective 和 DP `main_grad`/Reduce-Scatter 顺序。要引入独立 stream，必须先明确 Saved Tensor、TE workspace、RNG 和参数生命周期的跨流依赖并使用事件同步。

## 7. 附录：数据溯源与档案索引

- 测试与开关：`Megatron-LM-layerwise/tests/unit_tests/tensor_parallel/test_cross_microbatch_prototype.py`，通过显式 pytest 调用运行；默认训练代码没有新增开关或调度改动。
- 测试命令：`P8_RESULT_JSON=../qwen3_bench/results/proactive_cross_microbatch_20261119/p8_metrics_final.json python -m torch.distributed.run --nproc-per-node 1 -m pytest tests/unit_tests/tensor_parallel/test_cross_microbatch_prototype.py -q -s`。
- 日志：`p8_test_final.log`，1 passed；setup 的数据下载错误与该测试无关，完整日志保留。前一次运行见 `p8_test.log`。
- 机器数据：`p8_metrics_final.json` 的 `gpu_ms`、`cpu`、`counts`、`events`、`memory`、`peak_allocated`、`peak_reserved`；事件输出的 `output_bytes` 和 `saved_tensor_bytes` 对应缓存大小。
- 原有工作树快照：`preexisting_tracked.patch`、`preexisting_status.txt`、`preexisting_distributed_test.py`；P7 设计见 `DESIGN.md`；基线提交 `0765127eb4f2273b9aada48a24f285abec53b843`。
