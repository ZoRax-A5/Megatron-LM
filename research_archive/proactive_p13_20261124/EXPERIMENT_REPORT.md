# P13 · TP Stream 等待依赖与并行配置窗口初查

## 1. 核心摘要（Executive Summary）

本阶段完成真实 Qwen3-4B 训练的 CUDA Event 跟踪与三个并行配置的单步诊断。TP2 的 All-Reduce 提交范围内观测到 NCCL Stream 的 Event record 与另一 Stream 的 wait，说明相关 GPU Stream 存在事件依赖。**本次证据尚不足以证明某个 Replay 能在该等待之前合法提交，也不足以确定关键路径收益上界，因此 TP 并发 PoC 为 NO-GO；P13 为部分完成。**

- TP2/DP4/PP1、TP4/DP2/PP1、TP2/DP2/PP2 均完成 3 个训练 step，第 3 步单独采集 CUDA Event 与 kernel trace。
- TP4 的 All-Reduce GPU 驻留明显更长，但驻留包含 rank 等待。仅凭它不能认为有更多可隐藏的关键路径时间。
- PP2 引入不同的流水线顺序和跨 stage 依赖。本次通用观察器未完整标记 PP micro-batch 身份，不能据此给出跨 stage 的 Replay 候选或正确性结论。
- 所有候选的 TP 集合通信顺序、输入就绪与计算 Stream 等待链仍需一一关联，完成前不进入实现。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 模型 | Qwen3-4B 架构，BF16/TE，Seq2048、MBS1、GBS32；第 1、2、18 层 QKV 按需重计算 |
| 并行 | TP2/DP4/PP1，TP4/DP2/PP1，TP2/DP2/PP2，单节点 8×A100 80GB |
| 优化器 | Distributed Optimizer，原生异步 DP Reduce-Scatter，4000 万元素 bucket |
| 权重 | TP2/DP4 使用历史转换权重；TP4 与 PP2 使用固定 seed 随机初始化 |
| 观测 | NSYS CUDA/NVTX，启用 CUDA Event device tracing；仅第 3 步，诊断运行不作为无 profiler 性能结果 |

TP2 trace 用 NVTX 标出 All-Reduce API 区间，按 CUDA correlation ID 关联 NCCL kernel，再用 event ID 与 event sync ID 关联 record/wait。该归因要求 record 与 wait 均在对应 API 范围内，能确认事件关系。它不包含对最终依赖图的完整求解。

## 3. 正确性与观测完整性验证

三次训练命令退出码均为 0，无跳步或 NaN。这里没有重新运行全 rank 梯度和参数比较；TP2/DP4 的数值结论沿用 P11/P12，TP4 和 PP2 仅证明诊断训练可运行。三种配置的随机权重、micro-batch 数量和流水线行为不同，因此数据用于寻找窗口，不能直接比较模型质量或训练吞吐。

TP2 的 NVTX 归因产生 9,568 条 All-Reduce 范围内 Event record/wait 匹配，其中 9,544 条使用 NCCL Stream 39 到 Stream 151。其余 24 条属于不同进程组或 stream。Event 被重复使用，现有匹配没有证明每条 record 唯一对应哪个 Kernel，也没有证明目标 QKV 输入与该等待之间的依赖。不能将该计数当作独立候选窗口数量。

## 4. 性能评估与定量测量

各列为诊断第 3 步 8 rank 的中位数及范围，n=1 次采集。GPU span 为该 rank 首末 GPU kernel 间隔。All-Reduce union 为 rank 内 All-Reduce kernel 区间并集，包含等待与 profiler 干扰。

| 配置 | GPU span ms | All-Reduce kernel 数/rank | All-Reduce union ms | 单 Kernel 中位时长 ms | DP RS union ms |
|---|---:|---:|---:|---:|---:|
| TP2/DP4/PP1 | 1995.8 [1992.0, 2004.2] | 1198 | 308.0 [165.6, 443.2] | 0.111 [0.108, 0.118] | 46.6 [32.7, 75.5] |
| TP4/DP2/PP1 | 3771.1 [3768.9, 3773.1] | 2390 | 1517.3 [1036.1, 1644.9] | 0.404 [0.212, 0.465] | 22.1 [13.6, 30.9] |
| TP2/DP2/PP2 | 2407.3 [2405.2, 2410.6] | 1198.5 [1174, 1223] | 381.9 [339.2, 510.4] | 0.112 [0.109, 0.119] | 54.5 [12.8, 100.2] |

- 这些指标受 profiler 和不同权重初始化影响，不能作为性能提升或关键路径的量化结论。
- TP4 有更长的通信驻留，PP2 的两个 pipeline stage 具有不同的计算及通信工作量。

## 5. 深度分析与机制探讨

P12 对 TP2 的末 micro-batch Backward 找到 717 个按层和 TP 组枚举的正区间，其中 12 个可以在对应两 rank 的共同区间中容纳基线 Replay。没有一个固定的候选 ordinal 可让全部 4 个 TP 组同时容纳完整 Replay。这些是时间区间筛选的乐观上界。

TE 源码中 Forward 的 All-Reduce 使用同步调用，部分 Backward dgrad All-Reduce 使用异步调用与后续 Work 等待。新采集证实 API 内存在 event 同步，但目前无法从该关联证明特定依赖输入的生产者已经完成，或计算流能绕过已排入的等待。在这个边界之前发起 Replay 可能改变同组 collective 的排队顺序。

PP2 的 stage 边界还需要 P2P send/recv 的 event 依赖和每个 micro-batch 的实际身份。当前 PP2 的 NVTX 标记不能提供可靠的 Replay 生命周期，不能把空闲区段视为可用窗口。

## 6. 局限性、决策建议与后续工作

本轮仅有一个诊断步，没有硬件资源计数器或不带 profiler 的重复性能对照。TP4/PP2 未装载与 TP2 相同的预训练权重。GPU 驻留与关键路径暴露不是同一指标，不能用差值估计收益。

建议下一步先增强观察器，按 CUDA Event record 与 wait 的精确关联、通信组、micro-batch 和 layer 标识构建完整的跨 Stream 依赖图。PP2 还需明确 P2P 收发和流水线 stage 身份。只有找出所有目标 rank 一致满足输入就绪、等待顺序及消费截止点的实际窗口，才能开展单任务 PoC 和全 rank 正确性验证。

## 7. 附录：数据溯源与档案索引

| 内容 | 数据 |
|---|---|
| 基线 | `da39ca11384b97e7c057b994179078b0784aa6da`；P10–P12 正式报告位于 `research_archive/` |
| 配置和日志 | `p13_4b_C0_trace_conn1_{tp2_event,tp4,pp2}/manifest.json`、`train.log`、`status.json` |
| 逐 rank 统计 | 各 trace 目录 `aggregate.json`；全局摘要 `summary.json`；分析脚本 `aggregate.py` |
| Event 关系 | TP2 trace 目录 `event_wait_attribution.json`；归因脚本 `trace_events.py` 与 `analyze.py` |
| 完整 NSYS 数据 | 工作区 `qwen3_bench/results/proactive_p13_20261124/p13_4b_C0_trace_conn1_{tp2_event,tp4,pp2}/trace.nsys-rep` 与 `trace.sqlite`，未上传 |
| 执行脚本 | `poc.py`、`poc_entry.py`；P12 候选证据见其报告附录 |
