# P14 · TP Stream 依赖重建与 Replay 可调度性

## 1. 核心摘要（Executive Summary）

**P14 为部分完成，下一阶段 TP Replay PoC 为 NO-GO。** 对 P13 的 Event 数据按记录代次重新审计，并为 TP2 的一个 Backward All-Reduce 在两个 rank 上建立了 Kernel、Event record 和 wait 的条件关联。通信输入生产者、结果首次消费者以及候选 QKV 输入与通信结果的张量依赖仍未被唯一确定，因此不能给出经过验证的全 rank 提交窗口或关键路径收益。

- P13 三份 trace 的 `(pid, eventId, eventSyncId)` record 键均无重复。仍有部分 wait 缺少可匹配 record，必须保留未知状态。
- TP2 样例的通信 Kernel 提交先于同 Stream 上的 Event record，另一 Stream 随后提交 wait。Host 提交时间与 GPU Kernel 时间分开记录。
- TP4 的 Event 复用审计可应用同一方法，尚未完成四 rank 的通信输入与消费依赖链。PP2 新增了 stage、层、调用次序和 P2P 范围，但消息与 micro-batch 的身份仍待证明。
- 本轮没有改动 Megatron 训练语义，也未开展主动 Replay 性能实验。下一步先补齐张量和消息身份，再决定是否进行单任务 PoC。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 数据 | 复用 P13 的 TP2/DP4/PP1、TP4/DP2/PP1 与 TP2/DP2/PP2 NSYS SQLite |
| 新采样 | PP2，Qwen3-4B 结构，随机初始化，BF16/TE，Seq2048、MBS1、GBS32，3 步，仅采第 3 步 |
| 硬件与工具 | 单机 8×A100 80GB，NSYS CUDA Event tracing、NVTX |
| 候选 | 最后 micro-batch 第 1 层 QKV，TP2 两 rank 的第一个 Backward All-Reduce 样例 |

Event 审计按完整键收集所有 record，核查同键重复和 runtime API correlation。只有 record 与 wait 均能唯一关联到 API，且 host record 完成不晚于 wait 提交，才标为可匹配。将 NCCL Kernel 关联到 record 时，还要求同一进程、同一 Stream、相邻 record 之间只有一个可关联的 GPU Kernel。其余情况标为未知。提交顺序是 host 证据，GPU 时间由 kernel trace 单独给出。

## 3. 正确性与观测完整性验证

| P13 数据 | Event record 键重复 | 匹配且 host 顺序明确的 Stream wait | 不可匹配或有歧义的 wait | 生产 Kernel 无法唯一确定的已匹配 wait |
|---|---:|---:|---:|---:|
| TP2 | 0 | 60,736 | 996 | 50,240 |
| TP4 | 0 | 117,632 | 837 | 97,632 |
| PP2 | 0 | 59,816 | 629 | 49,472 |

对本批 trace，P13 单值字典未遇到同键 record 覆盖。其选择“后续最近 record”的 Kernel 归因缺少唯一生产者约束，不能用作已证明的 Kernel 依赖。已匹配 wait 可有多个消费者，按 wait 条数统计。

PP2 新观察器的每 rank Forward 和 Backward 调用计数均为 16。Forward 标识与 QKV 范围嵌套检查通过，stage 0 的层为 1–18，stage 1 的层为 19–36。P2P 记录提供 stage、调用类型及调用前计数，但发送/接收 payload 与 micro-batch ID 的对应、Backward 图身份尚未验证。诊断运行退出码为 0，不等于完整梯度正确性验证。

## 4. 性能评估与定量测量

选定样例为 rank 0/1 的同一 TP2 组，最后 micro-batch Backward 中各自第一个可关联 All-Reduce。单次诊断数据如下。

| rank | Host All-Reduce API 时长 | NCCL Kernel GPU 时长 | 原 QKV Forward 完成至 Kernel 开始 | Kernel 完成至原按需 Replay 开始 |
|---:|---:|---:|---:|---:|
| 0 | 0.391 ms | 0.132 ms | 101.081 ms | 173.865 ms |
| 1 | 0.112 ms | 1.097 ms | 100.114 ms | 173.862 ms |

这两个时间区间分别来自 CPU API 与 GPU Kernel，不能互相替换。候选 QKV 原始 Forward 已在上述 Kernel 前完成，按需 Replay 在其后发生。事件时间顺序本身不能证明可在 wait 前插入 Replay。

## 5. 深度分析与机制探讨

在 rank 0 样例中，NCCL Kernel 的 host 提交 correlation 为 434227，同 Stream 39 后续 Event record correlation 为 434229，Stream 151 的 wait correlation 为 434230。Event 使用同一 `eventId=54` 与 `eventSyncId=11041`。范围内唯一生产 Kernel 的提交、record 和 wait 顺序明确；Kernel 的实际 GPU 执行晚于这些 host 提交。这是一次具体事件链，并不证明通信输入的生产 Tensor 或后续首次消费 Tensor。

候选第 1 层 QKV 的 checkpoint 输入在原始 Forward 已存在，但 P13 捕获的是 QKV 输出计算 Kernel，缺少该张量的生产者、storage 身份及与所选 All-Reduce 输出的直接依赖记录。通信结果首次消费的位置也未能唯一绑定。两 rank 各自的第一个可关联调用还需要结合 communicator 序号和跨 rank 调用语义，才能证实它们是同一次 collective。上述边均标为未知。

P12 对三个候选 QKV 的乐观原 Replay Kernel 总量为 0.50–0.68 ms/rank；单图 allocated 增量约 22 MiB/rank。这是消除原 Replay 计算的理想上界，无法由 All-Reduce Kernel 驻留时间推导出可隐藏的关键路径时间。TP4 同键审计结果支持继续使用同一检查方法，但缺失输入/消费者链，不能比较可实现收益。

## 6. 局限性、决策建议与后续工作

本轮只对 TP2 一个样例做条件关联，并未建立完整 GPU Event DAG。Event 数据中的不可匹配项可能来自截取边界、异步线程或导出缺失，当前不能确定原因。没有证据证明候选在所有 TP rank 上可合法提交，也未测得该窗口在 step 关键路径上的暴露量。

下一阶段应先在选定通信调用采集输入/输出 Tensor 的 storage 身份、layer、communicator 序号与消费算子，联合 CUDA Event 关联验证跨 rank 同一次 collective。PP2 需要在调度器持有真实 micro-batch ID 的位置标记 send/recv payload 与 backward 图身份，确认同一消息两端匹配。只有这些检查通过，且计算得到有实际意义的关键路径收益上界，才进入 Replay PoC。

## 7. 附录：数据溯源与档案索引

| 证据 | 文件 |
|---|---|
| 基线与历史 | 分支基线 `12ab23e2c129f8d1cc68f6110bd4c369b9fc3d0e`；P12/P13 报告与脚本位于 `research_archive/` |
| Event 审计 | `event_audit.py`、`event_audit.log`，`event_audit_summary.json` 保存汇总，完整 `*_event_audit.json` 位于工作区 |
| TP2 样例 | `sample_graph.py`、`sample_graph.json`；每条记录包含 host API、GPU kernel、Event 和候选 QKV 时间，未知边显式列出 |
| PP2 新采样 | `poc.py`、`poc_entry.py`、`pp_audit.py`、`pp2_audit.json`；配置与日志位于 `p14_4b_C0_trace_conn1_pp2/` |
| 原始轨迹 | 工作区 `qwen3_bench/results/proactive_p13_20261124/` 的三份 `trace.sqlite` 与 `trace.nsys-rep`，以及 `qwen3_bench/results/proactive_p14_20261124/p14_4b_C0_trace_conn1_pp2/` 的新 trace。大型文件未上传 |
