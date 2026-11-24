# P11 · 原生 PP1 的通信感知 QKV Replay

## 1. 核心摘要（Executive Summary）

**P11 功能 PASS。整步加速未得到可重复证据。** 原生 PP1 可以在最后一个 micro-batch 的 Backward 内，利用已提交的异步 DP RS 提前执行第 1 层 QKV Replay。两种触发点均通过正确性验证。第 36 桶是本轮的优先实验方案，因为它在采集步缩短了缓存驻留并覆盖了全部 rank 的 RS 窗口。

- 第 36 桶方案在 8/8 rank 上观测到真实 GPU Kernel 交集，范围 0.238–0.337 ms。
- 全部 8 rank、连续 3 步的完整梯度分片、更新参数及 loss 与 C0 一致。每个成功实例执行 **1 次原始 Forward、1 次主动 Replay、0 次 Backward Replay**。
- 晚触发的 GPU Replay 完成至消费中位数为 1.387 ms，对照 C2 为 152.512 ms。整步 allocated 峰值比 C2 低约 5.15 GiB。
- 晚触发对 C0 的三轮整步时间差为 +7.0、-1.6、+40.6 ms。现有数据支持保留原型用于调度研究。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 模型与序列 | Qwen3-4B 历史转换权重，BF16/TE，Seq2048、MBS1、GBS32，固定 seed 1234、mock tokens |
| 硬件与并行 | 8×A100 80GB，TP2/DP4/PP1，SP 关闭，每 rank 8 个 micro-batch |
| 优化器与通信 | AdamW Distributed Optimizer，原生异步 DP Reduce-Scatter，bucket 设定 4000 万元素 |
| 重计算工作量 | 第 1、2、18 层 QKV，每步每 rank 288 次原始 QKV Forward、24 次 Replay |
| C0 | 原生 PP1 + 按需 Replay |
| C1-首桶 / C1-第36桶 | 原生 PP1，在最后一个 micro-batch 的第 1 层 QKV 实施主 Stream Replay，分别在第 1、第 36 个 RS 提交后触发 |
| C2 | P10 双 Forward 窗口 + 早期同 Stream Replay，并保留相同的最后 micro-batch 首桶触发目标 |
| 测量 | 3 步全量正确性，独立 NSYS/CUDA Event 采第 3 步。整步性能每模式运行 3 次，每次 12 步，前 5 步预热，轮换执行顺序 |

原始 QKV 的首个 GPU kernel 提供输入就绪的保守上界，C0 的按需 Replay 首个 kernel 标记原消费位置。提交窗口使用真实 RS kernel 区间，并裁掉 host 提交尚未返回的部分和消费截止点之后的部分。并发统计使用同一 GPU 上两类实际 kernel 区间的交集。CUDA Event 另外测量提交 Replay 到消费之间的 stream 距离，allocated 与 reserved 分别按每步采集峰值。

## 3. 正确性与观测完整性验证

| 验证 | 覆盖 | 结果 |
|---|---|---|
| 完整数值 | C0、C1-首桶、C1-第36桶、C2，8 rank × 3 步 | 每个候选模式 48 组全量指纹均与 C0 一致，逐 rank loss 完全相同 |
| 目标执行次数 | 每个已预取实例 | Forward / 主动 Replay / Backward Replay 为 **1 / 1 / 0**，命中、消费、释放各一次 |
| PP1 顺序 | C0 与两个 C1 模式 | 每 rank 保持 8 次 `Forward(m) → Backward(m)` 交替 |
| 原生通信 | C0 与两个 C1 模式 | 实测 37 次异步 DP RS/rank，均由最终 Backward 发起。NSYS 中每个 micro-batch 的模型 TP kernel 提交名称序列与 C0 完全一致 |

已归约梯度分片在优化器更新前读取，模型参数在更新后读取。每 rank 每步分别检查 502,873,984 个梯度元素和 2,011,495,936 个参数元素。指纹包括张量名称、形状、dtype 和完整字节。

默认关闭的实验参数只在原生 PP1 最后一个 micro-batch 建立第 1 层 QKV 的 proactive 句柄。host 触发点位于普通 DDP 梯度就绪钩子中，在原生异步 Reduce-Scatter 提交返回且退出 coalescing 后运行。触发作用域在 Backward 返回时解除，首步 DDP 桶就绪校准仍走按需路径。目标已消费、尚未遇到触发桶或首步未触发时保留按需执行。输入或参数版本错误由现有 checkpoint 校验终止。

P10 已发现 Saved Tensor 装饰钩子会改变 TE 反向取出的参数包装，并影响梯度就绪通知。目标 Replay 关闭该统计钩子。Prefetch 使用已有的 GradMode 与 RNG 保存/恢复，仅构造前向图。诊断运行逐次检查 CPU RNG、CUDA RNG 和 Megatron RNG tracker 在提交前后相同。目标是 SP 关闭的 BF16 TE 融合列并行投影，前向 Replay 本身不发起 TP collective。梯度仍由原 checkpoint Backward 传播。

## 4. 性能评估与定量测量

每组独立启动 3 次，单次 12 步。每轮先取第 6–12 步整步时长的中位数，再比较三个独立运行。显存为每轮所有 rank 的整次峰值最大值。

| 模式 | 整步中位数 ms［三轮范围］ | 最大 allocated GiB［三轮范围］ | 最大 reserved GiB［三轮范围］ |
|---|---:|---:|---:|
| C0 | 1632.9［1628.2, 1641.0］ | 22.305［22.303, 22.309］ | 24.184［23.826, 24.322］ |
| C1-首桶 | 1651.3［1643.9, 1655.7］ | 22.309［22.304, 22.309］ | 24.266［23.877, 24.391］ |
| C1-第36桶 | 1648.0［1631.3, 1668.8］ | 22.304［22.303, 22.305］ | 23.869［23.848, 24.059］ |
| C2 | 1719.9［1715.6, 1720.9］ | 27.457［27.456, 27.458］ | 29.355［29.303, 29.357］ |

| 与同轮 C0 对比 | 三轮整步差值 ms | 中位差 ms | 中位相对差 | 最大 allocated 差值中位数 |
|---|---|---:|---:|---:|
| C1-首桶 − C0 | +10.3 / +11.0 / +27.5 | +11.0 | +0.67% | +4.11 MiB |
| C1-第36桶 − C0 | +7.0 / -1.6 / +40.6 | +7.0 | +0.43% | -1.30 MiB |
| C2 − C0 | +74.6 / +88.0 / +91.7 | +88.0 | +5.39% | +5275.65 MiB |

C1-第36桶的整步变化为 -1.6 至 +40.6 ms，中位差 +7.0 ms。C1-首桶的中位差为 +11.0 ms。晚触发相对 C2 降低峰值 allocated 中位数 5.153 GiB，主要来自保留原生 PP1 调度。C1 与 C0 的峰值接近。每个 C1 运行在稳态的每 rank 每步只持有一个主动图，首步按需回退，无跳步或 NaN。

## 5. 深度分析与机制探讨

### 原生 C0 候选窗口

| 层 | 就绪上界到按需 Replay 的中位数 | 已测试或候选提交点 | 提交返回到按需 Replay 的中位数 | 各 rank 中最短的 RS 可覆盖区间 |
|---|---:|---:|---:|---:|
| 1 | 277.52 ms | 第 1 桶 / 第 36 桶 | 160.18 / 1.65 ms | 0.727 / 0.902 ms |
| 2 | 271.77 ms | 第 1 桶 / 第 35 桶 | 155.92 / 1.64 ms | 0.727 / 0.911 ms |
| 18 | 167.31 ms | 第 1 桶 / 第 19 桶 | 93.41 / 1.70 ms | 0.727 / 0.936 ms |

第 1、2、18 层都在最终 Forward 保留了输入。现有 BF16 TE QKV Replay 的前向计算读取已保存输入和参数，输出梯度只在后续消费时需要。目标不能依靠其自身梯度所属的桶触发。当前选择第 1 层，其首桶和第 36 桶的区间分别代表早触发和晚触发。第 2、18 层保留为候选，尚未实施。C0 的第 1、2 层按需 Replay 与 DP RS 交集均为 0。C0 的第 18 层按需 Replay 已在 6/8 rank 上与较早梯度桶存在交集，最大 0.340 ms，因此第 18 层能否从进一步移动受益，需要单独做对照。

### 实测 Kernel 交集与缓存寿命

| 模式 | 观察到交集的 rank | 目标 Replay∩RS，ms［跨 rank 范围］ | Replay 完成至消费，GPU ms［跨 rank 范围］ | 目标桶 RS，GPU ms［跨 rank 范围］ |
|---|---:|---:|---:|---:|
| C1-首桶 | 6/8 | 0.319［0, 0.321］ | 168.246［136.246, 183.921］ | 31.752［0.723, 47.551］ |
| C1-第36桶 | 8/8 | 0.333［0.238, 0.337］ | 1.387［1.377, 1.408］ | 1.218［0.806, 1.273］ |
| C2 | 6/8 | 0.313［0, 0.319］ | 152.512［136.323, 205.483］ | 18.604［0.718, 70.657］ |

表中为第 3 步的跨 rank 中位数。C1-首桶在 rank 2、3 的目标区间没有实际交集，GPU 上的 RS 分别提前约 0.98、1.13 ms 结束。输入依赖和消费期限在这两个 rank 均满足。C1-第36桶的目标 Replay 使用当前 Autograd 计算 Stream 35，C0 Backward 原本便使用此 Stream，原始 QKV Forward 在 Stream 7，NCCL RS 在 Stream 43。原型没有创建 CUDA Stream，CUDA Event 验证 Replay 完成、消费、释放的顺序。

两种 C1 的目标缓存 allocated 局部增量均为 22.01 MiB/rank。晚触发将 CPU 缓存驻留中位数从首桶的 169.65 ms 降至 3.09 ms。与同一轮 C2 对照相比，晚触发的 GPU 完成至消费间隔缩短约 99.1%。时间差变小，单个 Replay 图的临时体积保持一致。

第 36 桶方案的目标 Replay kernel 时长中位数为 0.333 ms，C0 对应按需 Replay 为 0.168 ms。并发时的延长与资源竞争相符。不同 rank 的 RS 到达偏差、一次采样及缺少硬件计数器使显存带宽与 SM 竞争的贡献无法分离。

## 6. 局限性、决策建议与后续工作

NSYS 每模式只捕获一个预热训练步，用于证明结构性顺序和真实 Kernel 交集。整步时间从独立无 profiler 运行取得。长序列、其它拓扑、随机算子配置及 GPU SM/显存带宽硬件计数器不在本轮测量范围。固定目标绑定本次可预测的本地桶提交顺序。跨模型或桶设置应用前需重新确认输入与截止点。

继续研究时优先使用原生 PP1 的晚桶触发，进一步检查桶分组和目标消费之间的稳定性，区分等待被隐藏的时间与 Replay 在 NCCL 同时执行时增加的时延。本轮没有足够性能依据开启默认使用。

## 7. 附录：数据溯源与档案索引

| 项目 | 文件 |
|---|---|
| 版本 | 基准提交 `53c3660bab6a2396f061e5f5a6d7d80555a2cd9a`，`source_53c3660ba.tar.gz`，原 P10 未提交补丁 `p10_preexisting.patch`，最终工作树差异 `p11_worktree.patch` |
| 设计与运行 | `DESIGN.md`、`poc.py`、`poc_entry.py`、各用例 `manifest.json`、`git_initial.txt` 和 `git_final.txt` |
| 原生候选窗口 | `p11_4b_S0_trace_conn1/` 的 `trace.nsys-rep`、`trace.sqlite`、`tasks.json`、`nvtx_attribution.json`，`candidate_windows.json` 由 `windows.py` 生成 |
| 数值验证 | `*_verify_conn1/` 的 `exact_rank*.json` 和 `train.log`，`comparison.json`；逐 rank 全量字节指纹由 `exact_entry.py` 生成 |
| 并发与性能 | `*_trace_conn1/` 的完整 NSYS 报告和 SQLite，`*_perf_conn1_r*/` 的 `train.log`、`rank*.json` 和配置，`summary.json` 的 `timeline`、`runs`、`paired`；分析入口为 `analyze.py` 和 `summarize.py`。Kernel 片段图由 `plot.py` 生成。TP 提交签名审计见 `tp_order_audit.json` |
| 设备与工具 | `gpus.csv`、`topology.txt`、`nsys_version.txt`、`source_sha256.txt` |
| P9/Checkpoint 回归 | `unit.log`，现有两份单 GPU 用例共 19 项通过。 |
