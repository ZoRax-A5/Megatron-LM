# P12 · QKV Replay 的通信隐藏收益与候选窗口

## 1. 核心摘要（Executive Summary）

**P12 对照与机制验证 PASS。以现有 DP 固定桶方案继续扩大 Replay 数量，性能判断为 NO-GO。** C1 将 QKV 前向 Replay 与已被原生 Backward 覆盖的早一桶通信并发。仍然暴露的终端 DP RS 约 3.05–3.08 ms，来自必须等第 1 层梯度的末桶。第 36 桶从未进入这段尾部。

- 原位置单个 QKV Replay 只占约 0.168 ms。并发时同一 QKV 的 Kernel 并集升至约 0.326 ms，观察到的交集不能直接计为收回的时间。
- 五轮独立运行中，C1 相对 C0 的整步差值范围为 -75.1 至 -7.2 ms，相对空钩子为 -51.0 至 +10.7 ms。C1 五轮都快于 C0，但相对 Cnoop 只有两轮更快，中位差为 +1.7 ms。该幅度无法由单个 Replay 的计算消除解释。
- 第 35 桶把 Replay 完成至消费的 GPU 距离拉长至 5.97 ms，第 36 桶为 1.39 ms。两处缓存成本都是 22.01 MiB/rank，每 rank 最多同时 1 图。
- 8 rank 完整数值通过。离线考察第 1、2、18 层之后，未开展多任务 GPU 运行或 TP/SP 改造。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 硬件 | 8× A100 80 GB，单节点 |
| 软件 | PyTorch 2.7.0a0+79aa17489c.nv25.4、TE 2.2.0+c55e425、NCCL 2.26.3+cuda12.9、NSYS 2025.2.1 |
| 模型与训练 | Qwen3-4B 转换权重，BF16/TE，Seq2048、MBS1、GBS32，mock tokens，固定种子 |
| 并行、优化器 | TP2/DP4/PP1，原生逐 micro-batch Forward/Backward，SP 关闭，AdamW Distributed Optimizer |
| 通信与重计算 | 原生异步 DP Reduce-Scatter，4000 万元素梯度桶，选择 QKV 层 1、2、18 |
| C0 | 原生按需 checkpoint |
| Cnoop | 使用第 36 桶触发路径和相同目标句柄。钩子返回后仍在原 Backward 位置 Replay |
| C1 | 第 36 桶提交后提前重放第 1 层 QKV |
| Clate | 另一个晚触发点，第 35 桶。它比 C1 早一个桶，仍接近目标 Backward |
| 测量方法 | 每组 3 步全量正确性。每组单独采第 3 步 NSYS/Events。每模式 5 次独立无 Profiler 运行，每次 10 步，前 5 步预热，轮换运行顺序 |

通信驻留指 NCCL Kernel 的 GPU 时间区间并集，包含 rank 等待。为定位优化器前的 DP 屏障，在最后一次 Backward 的非 NCCL 计算及其模型 TP Kernel 完成后，测量到最后一个 DP RS 完成的剩余墙钟时间。该值是 **DP 终端暴露窗口**。它不计入中途资源竞争或隐藏的同步依赖，不能当作可回收加速。没有采集完整的 CUDA event 依赖图，报告用实际 Kernel 时间戳、原生 `finish_grad_sync` 的等待链和优化器启动边界交叉核对。

CPU 提交与等待通过 NVTX/CUDA API 归因。Kernel 并发只按同一 GPU 实际执行的区间交集计算。单次移动的乐观计算消除上界取 C0 原重放 Kernel 执行时间，条件是假设它原处于关键路径且新位置的计算完全免费。该上界用于判断量级，不能界定所有可能的调度或资源编排策略。

## 3. 正确性与观测完整性验证

Cnoop、C1 和 Clate 均通过相对 C0 的 8 rank、连续三步完整梯度与更新参数对照，每候选模式 48 次完整张量指纹一致，逐 rank 原始 loss 标量一致。每 rank 每步覆盖 502,873,984 个归约有效梯度分片元素和 2,011,495,936 个参数元素。指纹覆盖张量名、形状、dtype 和全部原始字节。

所有模式保持相同的八次序列内 Forward/Backward、原始 QKV Forward 工作量及按层 Replay 总次数。新增实验没有关闭异步梯度通信。Cnoop 量的是实验触发路径的全部增量，包含目标主动句柄、查找、调用及回退的代价。它不能单独识别每行 Python 或某次监测指令的耗时。

NSYS 审计验证各组逐 rank 的 TP 模型通信 Kernel 提交签名一致，均有 37 次原生异步 DP RS，桶顺序与 C0 一致。两个主动模式的目标执行次数为 1 次原 Forward、1 次主动 Replay、0 次 Backward Replay。Cnoop 命中钩子但没有提前产生 Kernel，目标仍在原位置重放。三组实验路径首步按需回退，后续所有目标均按规划触发并释放。

## 4. 性能评估与定量测量

无 Profiler，每轮取第 6–10 步的整步中位数。峰值指标先对每轮的全部 rank 取最大值，再汇总五轮。两组运行的配对依据预先分配的轮次。

| 模式 | 整步中位数 ms［五轮范围］ | 峰值 allocated GiB［范围］ | 峰值 reserved GiB［范围］ |
|---|---:|---:|---:|
| C0 | 1653.0［1643.7, 1700.8］ | 22.305［22.303, 22.308］ | 24.064［23.836, 24.139］ |
| Cnoop | 1633.7［1624.0, 1677.9］ | 22.304［22.303, 22.308］ | 23.908［23.797, 24.332］ |
| C1 | 1626.9［1612.4, 1644.2］ | 22.305［22.303, 22.309］ | 24.109［23.988, 24.348］ |
| Clate | 1662.8［1626.7, 1679.2］ | 22.304［22.303, 22.309］ | 23.973［23.883, 24.383］ |

| 配对比较 | 五轮整步差值 ms | 中位差 ms | 中位相对差 | 峰值 allocated 中位差 |
|---|---|---:|---:|---:|
| Cnoop-C0 | -17.9 / +34.2 / +5.9 / -76.8 / -31.3 | -17.9 | -1.08% | -0.98 MiB |
| C1-C0 | -7.2 / -16.8 / -40.6 / -75.1 / -28.6 | -28.6 | -1.72% | -0.10 MiB |
| Clate-C0 | +11.4 / -16.3 / -26.3 / -33.1 / +14.2 | -16.3 | -0.99% | -0.74 MiB |
| C1-Cnoop | +10.7 / -51.0 / -46.5 / +1.7 / +2.7 | +1.7 | +0.10% | +0.88 MiB |
| Clate-Cnoop | +29.3 / -50.5 / -32.2 / +43.7 / +45.5 | +29.3 | +1.79% | +0.74 MiB |
| Clate-C1 | +18.6 / +0.5 / +14.3 / +42.0 / +42.8 | +18.6 | +1.13% | +0.33 MiB |

空钩子相对 C0 的轮次差值为 -76.8 至 +34.2 ms，中位差 -17.9 ms。诊断中仅包围空返回的 NVTX 范围中位数为 1.82 µs，该值不含注册、查找、目标图回退及测量之外的其它代码。整步对照的波动比微秒级开销大得多，无法从整步数据精确分离每个钩子的成本。C1 相对 C0 的配对中位差为 -28.6 ms，相对 Cnoop 为 +1.7 ms。Clate 相对 C1 中位差 +18.6 ms，两个位置的缓存局部代价均为 22.01 MiB/rank。数据保留了 C1 相对原路径的五轮同向变化，但空钩子对照未复现稳定的净收益。

## 5. 深度分析与机制探讨

### P11 时间线重算

P11-C0 的 DP RS 总驻留在不同 rank 为 45.77–106.92 ms，中位数 76.64 ms。与普通 Backward 计算或其 TP 并集相交的中位数为 48.36 ms。剩余部分还包含等待其它 rank、两个 launch 间隙和通信排队，不能全部计为可利用窗口。

重要的是参数更新前实际剩余的梯度通信。在 P11-C0，最后一个 RS 结束相对所有 rank 最后一次 Backward 计算/TP 完成的间隔为 **3.070 ms**，C1 为 **3.067 ms**。第 36 桶对两组的这段尾部贡献都为 0。第 37 桶位于目标 QKV 梯度之后，不能用它为生成自身梯度的 Replay 创造提前窗口。

第 1 层按需 Replay 的 Kernel 并集中位数为 0.168 ms，两个 Kernel 之间的缝隙约 0.001 ms。提前后总 Kernel 占用中位数变为 0.333 ms。在采集到的 C1 中，第 36 桶在 Replay 首个 Kernel 启动时剩余 0.240–0.681 ms。**观察到 8 个 rank 有交集，6 个 rank 完整覆盖两个目标 Kernel。** rank 6/7 的桶较早结束。原位置被消除的计算最多约 0.17 ms，目标桶自身在原调度中已被其余的 Backward 覆盖。重叠并未直接收回约 3 ms 的终端尾部。

### P12 对照的关键路径证据

| 模式 | 目标 Replay Kernel 中位数 ms | 目标与选定 RS 的实际交集 ms［各 rank 范围］ | Replay 完成至消费中位数 ms | 跨 rank 终端 RS 尾部 ms | 诊断 GPU step 包络中位数 ms |
|---|---:|---:|---:|---:|---:|
| C0 | 0.168 | 0.000［0.000, 0.000］ | 不适用 | 3.058 | 1980.2 |
| Cnoop | 0.168 | 0.000［0.000, 0.000］ | 不适用 | 3.045 | 1910.2 |
| C1 | 0.326 | 0.326［0.206, 0.339］ | 1.389 | 3.073 | 1925.4 |
| Clate | 0.327 | 0.327［0.151, 0.332］ | 5.966 | 3.080 | 1914.5 |

四组首个优化器 GPU Kernel 均晚于各自最后一个 DP RS 完成。C0 两者差值的跨 rank 中位数为 11.59 ms，此区间还经过梯度 finalize、后续同步和优化器准备。第 35/36 桶均在最终 Backward 结束前完成，因此其并发不能直接消除后面的末桶服务。若提前执行能压缩最后一桶提交前的关键计算，末桶仍可能更早启动。对本目标而言，单靠消除原 Replay 计算的理想改善量级仍约 0.168 ms。四组在梯度 finalize 中各有 37 次 CUDA 等待 API 提交/rank。诊断单步的 GPU 包络有显著波动，例如空钩子也出现了约 70 ms 的缩短。整步收益据无 Profiler 的配对结果判断。

新 trace 中 C1 和 Clate 各有 8/8 rank 看到交集，各有 6/8 rank 的两个目标 Kernel 完全落入同一个 RS。C1 在首个 Replay Kernel 开始时，目标 RS 的剩余时间为 0.208–0.959 ms，中位数 0.777 ms。C0 的目标 QKV API 入口至首次 GPU 执行为 0.343 ms 中位数，该数包含 CPU 构建和 CUDA 排队。两个 QKV Kernel 之间的 GPU 间隔仅约 0.001 ms。C1 在原位置消除的两个 Kernel 合计约 0.168 ms，但提前位置的计算仍在原生 Autograd 计算 Stream 上运行。

在 C0 的相应原生 Replay 计算 Stream 上，第 35/36 桶候选区间内最长的无 Kernel 区段中位数分别为 1.20 和 1.31 ms，几何长度足以容纳实测 Replay。这类区段还可能存在未记录的 CUDA 等待，不能用来证明所有间隙可调度。C1 与 Clate 的实测已验证这两个具体触发点的合法执行。

分开看，C0 的 RMSNorm 和 GEMM 的 Kernel 中位数分别为 33.76 和 134.54 µs，C1 为 58.93 和 269.97 µs。两类 Kernel 延长与 NCCL 同时占用 GPU 资源相符。该对照无法进一步断言是哪一种 SM 或内存资源造成的。

### 少量多任务候选的收益上界

| 末 micro-batch 目标层 | 最新观测可用梯度桶 | 该桶返回至原 Replay 中位数 | P12 C0 原 Replay 中位数 |
|---|---:|---:|---:|
| 第 1 层 | 36 | 1.671 ms | 约 0.168 ms |
| 第 2 层 | 35 | 1.658 ms | 约 0.168 ms |
| 第 18 层 | 19 | 1.668 ms | 约 0.168 ms |

候选筛选同时要求原输入已就绪、目标尚未消费，以及同 rank 真实 RS 窗口的剩余长度覆盖原始 Replay。第 18 层在 P11 采样中已有自然重叠，在 P12 C0 采样中为零，说明候选机会随 rank 到达变化。当前三个 Replay 的原计算全消除上界为每 rank 0.503–0.505 ms，P11 的基线估计为 0.503–0.676 ms。两图若同时驻留，按单图实测成本估计约 44.02 MiB/rank。

**多任务 GPU 验证 NO-GO。** 上述可移动点有助于扩大可观察的 Kernel 交集，但选出的桶均不能覆盖依赖目标梯度的终端 RS。低于 0.68 ms 的乐观计算上界小于本轮整步波动，缺少其能缩短关键路径的证据。依据阶段门槛，本轮停在离线候选验证。

### TP 与 SP

P12 C0 每 rank 观察到 608 个 Forward All-Reduce、584 个 Backward All-Reduce，并有 32 个 Forward 数据 Broadcast。单 All-Reduce 跨 rank/调用的中位数分别为 124.46 和 127.86 µs。模型通信总驻留的跨 rank 中位数为 310.83 ms，其中 80.32 ms 与普通计算 Kernel 有实际交集。剩余驻留包含运算等待和 rank 到达偏斜，没有因果依据把它全部算作暴露时间。

用实际 TP 组成员和相同 collective 顺序匹配两端，在末 micro-batch Backward 内取两个 rank 都已返回提交、都仍在执行通信、且目标仍有合法输入和消费期限的共同区间。枚举得到 717 个正区间，计数包含不同层和 TP 组组合。12 个区间可容纳所属 TP 组两端的基线 Replay，最长 0.198 ms。没有同一候选 ordinal 能让四个 TP 组都完整覆盖。模型 TP 同步还会把通信等待排入依赖计算 Stream，单凭这些区间不能认为同一 Stream 已能安全提前插入。

**TP 暂不比 DP 更值得直接实现原型。** 若继续调查，先在一个有共同窗口的高层 All-Reduce 上加最小 CUDA 等待依赖观测，确认低层已保存 QKV 输入独立于该归约输出，再评估能否把纯计算提交放在合法的 Stream 等待之前。保持原 collective 调用次序，具备 P11 同等级的全量梯度对照再扩展。SP 与细粒度 QKV checkpoint 现有配置互斥。适配需处理 Replay 自身的 AG/RS、保存输入的分片语义和跨 rank 次序，本轮仅保留此分析。

## 6. 局限性、决策建议与后续工作

每模式只采一个 NSYS 诊断步，用于确认调度、时间区间及依赖结构。不同模式的 GPU 轨迹会受到 rank 到达偏斜、GPU 时钟和 profiler 扰动影响。两组 trace 的整步墙钟差不能代替无 Profiler 性能测量。设备计数器未采集，SM 与 HBM 竞争没有被直接分离。

保留 P11 晚触发机制供后续研究，但不以增加 Replay 数量作为当前加速方向。后续若要投入，优先找能减少优化器实际等待的结构性变化，再在固定 batch 和通信参数下验证；同时比较 C0、Cnoop 和目标方案的 GPU 依赖与无 Profiler 整步数据。

## 7. 附录：数据溯源与档案索引

| 证据 | 文件 |
|---|---|
| 源码版本 | `53c3660bab6a2396f061e5f5a6d7d80555a2cd9a`、`source_53c3660ba.tar.gz`、`p10_preexisting.patch`、`p11_worktree.patch`、`git_initial.txt`、`git_final.txt`、`source_sha256.txt` |
| P11 输入 | 上一轮 `p11_4b_{S0,C1late}_trace_conn1/` 原始 NSYS 文件和归因数据。重算数据为 `p11_*_exposure.json` 以及 `p11_*_candidates.json` |
| 运行入口与环境 | `poc.py`、`poc_entry.py`、各用例的 `manifest.json` 与 `train.log`，`gpus.csv`、`topology.txt`、`nsys_version.txt` |
| 完整数值 | `comparison.json`，各 `*_verify_conn1/exact_rank*.json`，哈希记录由 `exact_entry.py` 生成 |
| 新时间线 | 各 `*_trace_conn1/` 的 `trace.nsys-rep`、`trace.sqlite`、`comm.json`、`tasks.json`、`nvtx_attribution.json`，以及 `p12_*_exposure.json`。归因、终端曝光和候选计算分别由 `analyze.py`、`critical_path.py`、`candidates.py` 产生 |
| 关键路径图 | `critical_path_timeline.png`，由 `plot.py` 从真实 GPU Kernel 时间戳生成。Replay 生命周期及通信提交审计为 `timeline_audit.json`，固定 trace 的计算 Stream 几何间隙见 `fixed_trace_slack.json`，finalize 等待 API 见 `cuda_grad_finish_wait_api.json`。 |
| 重复性能 | 各 `*_perf_conn1_r*/` 的日志、8 rank 记录及配置，结构化比较 `performance_summary.json`，生成器 `summarize.py` |
