# P10 · 通信窗口与独立 CUDA Stream 的 QKV Replay

## 1. 核心摘要（Executive Summary）

**P10 功能 PASS，独立 Stream 性能推广 NO-GO。** 在 Qwen3-4B 的最后一个 micro-batch，首个 DP 梯度桶提交后，可以提前执行第 1 层 QKV Replay，并与 NCCL Reduce-Scatter 形成真实 GPU kernel 并发。相同触发点的主 Stream 方案也能形成并发。独立 Stream 的三轮配对整步时间差有正有负，未显示稳定收益。

- 小模型三轮均复现 S1 比 S0 更快、S2 比 S1 更慢。4B 的首轮五模式对照中，原始 S0 保持最低整步时间和显存峰值。
- 现有 P9 调度的早期 Replay 与 DP RS GPU 交集为零。首个梯度桶后触发的 4B 固定目标在 S3 的 6/8 rank 上形成交集，最大 0.319 ms。
- Qwen3-4B 的 8 个 rank、3 步全量有效梯度和更新参数通过 SHA-256 对照，逐 rank loss 一致。目标计算图被 Backward 消费，未重复 Replay。
- 每个固定目标的局部 allocated 增量为 22.01 MiB/rank。4B 的整步显存成本主要来自双 Forward 窗口。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| GPU 与并行 | 单节点 8×A100 80GB，TP2/DP4/PP1，8 micro-batch/rank，MBS1，GBS32 |
| 软件 | PyTorch 2.7.0a0+79aa17489c.nv25.04，Transformer Engine 2.2.0+c55e425 |
| 小模型 | 两层 Qwen3 架构，Seq256，随机初始化，两个 QKV checkpoint |
| Qwen3-4B | 载入历史转换权重，Seq2048，三层选择 QKV checkpoint，分别为 1、2 和 18 |
| 优化器与通信 | BF16、AdamW Distributed Optimizer、原生异步梯度 Reduce-Scatter，bucket 设置为小模型 40 万和 4B 4000 万元素 |
| 模式 | S0 原调度按需，S1 双 Forward 窗口按需，S2 原 P9 同 Stream，S2-DP 与 S3 在相同桶触发点分别使用主 Stream 和独立 Stream |
| 对照采样 | P10.1 每模式三次启动，每次 30 步，前 5 步预热，启动轮次内变换顺序。时间线使用独立运行的第 3 步 |
| 指标 | 训练日志整步耗时，各 rank 进程内 allocated/reserved 峰值，NSYS CUPTI kernel 区间以及 CUDA Event 的提交与消费边界 |

通过 NVTX 与 CUDA API correlation ID 关联真实 GPU kernel，叶范围按提交线程归因。Backward 和整步的父范围额外纳入同进程 autograd 工作线程。通信归因区分梯度桶 Reduce-Scatter、模型 TP/SP collective、Replay 身份握手及步级其它 collective。GPU 并发使用相交的真实 kernel 区间的并集长度，避免重复计数。输入就绪以原始 QKV 首个 GPU kernel 的时刻作为保守上界。

## 3. 正确性与观测完整性验证

| 验证 | 覆盖 | 结果 |
|---|---|---|
| P10.1 小模型基线 | 每模式 3×30 步，每步 8 rank | 所有模式的逐 rank loss、QKV 执行次数、梯度桶触发边界一致 |
| 小模型固定桶原型 | S0、S2-DP、S3，8 rank × 3 步 | 逐 rank loss、全量归约梯度分片和参数 SHA-256 一致 |
| 4B 固定桶原型 | S0、S1、S2、S2-DP、S3，8 rank × 3 步 | 每候选模式 48 组完整张量指纹匹配，逐 rank loss 完全相同 |

每步在梯度 finalize 后、优化器更新前读取所有本 rank 拥有的已归约梯度分片，更新后读取全部模型参数。4B 每 rank 每步分别覆盖 502,873,984 个梯度元素和 2,011,495,936 个参数元素。指纹包含名称、形状、dtype 和全部原始字节。P9 的逐 rank 张量差异验证另外保留，用于审计先前的 S0/S1/S2 结果。

初次 PoC 暴露两条约束。第一步 DDP 还未建立梯度就绪计数，归约在 finalize 才提交，因此这一迭代使用按需 Replay。另一个问题出现在梯度同步的最后一个 micro-batch。用于估算存储大小的 Saved Tensor 钩子改变了 TE 反向取出的参数包装，DDP 少接到一次就绪通知。同 Stream 对照复现了问题，关闭目标 Replay 的存储统计钩子后，原生 DDP 检查恢复并通过全量对照。存储统计开关是此次 PoC 唯一的 Megatron 源码补丁。

## 4. 性能评估与定量测量

| P10.1 小模型 | 整步中位数 ms［三次运行范围］ | allocated MiB［范围］ | reserved MiB［范围］ |
|---|---:|---:|---:|
| S0 | 212.8［209.6, 215.8］ | 84.60［84.60, 84.60］ | 116［116, 118］ |
| S1 | 188.7［186.2, 193.9］ | 87.92［87.92, 87.92］ | 118［118, 118］ |
| S2 | 201.2［195.6, 202.7］ | 88.42［88.42, 88.42］ | 118［118, 118］ |

- 每个运行先取 25 个稳态步的中位数，再按模式比较三个独立运行。配对后，S1 相对 S0 的中位数为 −23.4 ms（−11.16%），三次均更快，范围为 −27.1 至 −18.9 ms。S2 相对 S1 中位数为 +12.5 ms（+6.62%），三次均更慢，范围为 +1.7 至 +16.5 ms。
- 调度顺序与原 P9 主动缓存分别增加峰值 allocated 3.32 MiB 和 0.50 MiB。各运行、各 rank 每步都执行 16 次原始 QKV 和 16 次 Replay。S2 中 6 次 Replay 提前执行，10 次在 Backward 按需执行。

### Qwen3-4B 对照

| 4B 模式 | 整步中位数 ms［步间范围］ | 相对 S0 | allocated 峰值 GiB | reserved 峰值 GiB |
|---|---:|---:|---:|---:|
| S0 | 1628.0［1609.4, 1664.6］ | 基线 | 22.303 | 23.758 |
| S1 | 1708.5［1692.7, 1739.7］ | +4.94% | 27.397 | 29.279 |
| S2 | 1716.9［1713.6, 1752.9］ | +5.46% | 27.458 | 29.285 |
| S2-DP | 1716.2［1707.3, 1731.3］ | +5.42% | 27.456 | 29.355 |
| S3 | 1727.0［1707.0, 1749.2］ | +6.08% | 27.456 | 29.379 |

表中各模式为一次 12 步运行，前 5 步预热，显存为 8 rank 整段最大值。4B 的 S1–S0 峰值 allocated 增加 5.095 GiB，S2–S1 再增加 62.41 MiB。S2-DP 与 S3 的追加目标图均增加局部 allocated 22.01 MiB/rank，其中 QKV 输出为 12 MiB。S3 目标缓存 CPU 驻留中位数为 174.64 ms，范围 145.27–222.12 ms。原 S2 的 72 个提前图中，每图局部增量均为 22.01 MiB，CPU 驻留为 114.02–251.85 ms。最多 3 个早期图同时驻留，最后一个 micro-batch 的追加目标图独立驻留。同一图的局部 allocator 差值与整步 allocator 峰值采用不同口径。

三种主动模式每 rank 每步都执行 288 次原始 QKV Forward 和 24 次 Replay。S2 的 9 个提前任务与 S0/S1 的按需任务对应。两个固定桶模式从第二步起各有 10 次提前 Replay 与 14 次按需 Replay，目标 checkpoint 均完成消费和释放。

| 固定桶配对运行 | S2-DP 中位数 ms | S3 中位数 ms | S3−S2-DP | 相对 S2-DP |
|---|---:|---:|---:|---:|
| 1 | 1716.2 | 1727.0 | +10.8 ms | +0.63% |
| 2 | 1719.9 | 1715.9 | −4.0 ms | −0.23% |
| 3 | 1722.0 | 1725.9 | +3.9 ms | +0.23% |

配对时间差中位数为 +3.9 ms，范围 −4.0 至 +10.8 ms。S3 相对 S2-DP 的最大 allocated 差值范围为 −0.84 至 +3.98 MiB，最大 reserved 差值范围为 −186 至 +24 MiB。可重复的结论是两个方案都能提交并消费同一个目标，独立 Stream 的附加成本落在本实验的步时波动内。

## 5. 深度分析与机制探讨

| 候选窗口 | 依赖与判定 |
|---|---|
| 原 P9 前三对 micro-batch 与 DP RS | 首个 DP RS 只在最后一个 Backward 发起。小模型最后一次主动 Replay 结束至首个 RS 有 86.21–93.71 ms 间隔，4B 为 628.42–639.58 ms。现有 S2 的 GPU 交集均为 0，且原目标的梯度早已计算 |
| 最后 micro-batch 已就绪的上层梯度桶与下层 QKV | 4B 的最后一个 micro-batch 中，层 1、2、18 的输入已在 Forward 保存。首桶提交返回至原重放位置之间，各 rank 实测留下 0.571–57.988 ms 的未完成 RS 区间。DQKV 依赖该层输出梯度，纯 BF16 QKV Replay 的前向只依赖保存的输入和参数。选择层 1 做固定范围原型 |
| 当前层 QKV 与它自己的梯度桶 | 该桶需要该层的梯度就绪。若提前执行 Replay，必须使用已经就绪的其它桶 |
| TP 模型通信 | SP 关闭时主要为 Forward 输出和 Backward 输入梯度的 All-Reduce，另有 Forward 数据 Broadcast。前驱 micro-batch 的 QKV 可在后继 micro-batch 的 TP 窗口前就绪。原 S2 的实际交集为零。该候选需要可重复的发起钩子和单独的 collective 顺序验证 |
| SP 通信 | 独立 SP 对照中，Forward 和 Backward 均看到 TP 组的 All-Gather 与 Reduce-Scatter。现有配置校验拒绝 SP 与选择性 QKV checkpoint 的组合，当前不可直接调度 |

原生异步 DP RS 与其它 GPU 计算在小模型的诊断单步内交集为 0–0.250 ms/rank，在 4B 为 28.42–59.71 ms/rank。4B 的 DP RS kernel 数为 37/rank，单 kernel 时长中位数 1.436 ms，rank 内时间并集为 35.30–117.34 ms。小模型为 2 个 kernel/rank，单 kernel 中位数 3.374 ms，范围 0.021–30.226 ms。长尾受 rank 到达偏斜和采集扰动影响，相关时间只用于定位诊断窗口。

模型通信 kernel 的位置和时长如下。计数均按一个 rank 的第 3 步汇总，时长是所有 rank 的单 kernel 中位数。

| 配置 | Forward 模型通信 | Backward 模型通信 |
|---|---|---|
| 小模型，无 SP | All-Reduce 64 个，70.9 µs | All-Reduce 40 个，36.3 µs |
| 4B，无 SP | All-Reduce 608 个，118.3 µs | All-Reduce 584 个，128.7 µs |
| 小模型，SP | All-Gather 40 个，38.7 µs。Reduce-Scatter 40 个，34.4 µs | All-Gather 80 个，14.2 µs。Reduce-Scatter 40 个，36.0 µs |

4B 的层 1、micro-batch 0/2/4 从输入就绪保守上界到整个 Backward 开始的各 rank 中位数分别为 237.65、216.66、212.63 ms。层 18 的相应中位数为 182.65、184.34、182.84 ms。对局部 host 提交后仍未结束的 TP All-Reduce，层 1 在上述三个 micro-batch 的最长连续交集的各 rank 范围分别为 0.124–13.517、0.629–6.377、0.129–2.483 ms。这些是候选区间。同一 collective 触发点能否在各 rank 稳定复现，仍需进一步确认。

**原型的同步与生命周期。** 输入生产方记录 Event，Replay Stream 等待该 Event。TE 2.2 的 GEMM workspace 为进程内共享资源，因此独立 Stream 在 Replay 前后与主计算 Stream 交接 workspace。首个 DP RS 已提交给 NCCL Stream，仍可独立执行。消费时再等待 Replay 完成 Event，跨 Stream 使用的 Tensor 通过 allocator 的 Stream 记录维持存活，反向输出同步回主 Stream 后进入原训练路径。

**实测并发。** S2-DP 在 7/8 rank 上观察到真实 kernel 时间交集，中位数 0.319 ms，范围 0–0.321 ms。S3 在 6/8 rank 上观察到交集，中位数 0.318 ms，范围 0–0.319 ms。rank 0 的 Replay 在 S2-DP 上使用 Stream 35，S3 使用 Stream 55，两组 NCCL RS 都在 Stream 43。两组的目标 Replay 都没有额外的 Backward Replay。图中仅显示真实 kernel 区间。

S3 的目标 Replay kernel 并集为 0.167–0.319 ms，首个 RS kernel 为 0.722–77.268 ms。两种流模式每 rank 均保持 37 个原生 DP RS。全步 DP RS 与所有计算 kernel 交集的各 rank 中位数，原 S2 为 57.04 ms，S2-DP 为 60.15 ms，S3 为 60.51 ms。该 trace 中有交集的 rank 的 Replay 约为 0.314–0.319 ms，未观察到交集的 rank 约为 0.167–0.168 ms。这个差异与 NCCL 和 GEMM 的 GPU 资源竞争相符。rank 到达偏斜、单步采集以及缺少 SM/显存带宽计数器，让这组数据无法单独确定竞争原因。

S3 新增的 workspace 交接 CPU API 范围中位数为进入前 22.58 µs、提交后 7.04 µs。反向流交接的两个范围中位数分别为 18.75 和 14.85 µs。这些是 CPU 标记内的提交时间。GPU 侧 Replay 提交结束到消费的 Event 距离为 143.63–220.71 ms，中位数 172.24 ms，包含其它 Backward 工作。

## 6. 局限性、决策建议与后续工作

这套 PoC 使用一个固定的桶、micro-batch 和 QKV 层。第一步的梯度桶顺序尚未建立，保留按需路径。固定目标的 TE BF16 QKV Replay 经 NSYS 检查只包含计算 kernel，PoC 对该目标关闭 TP 身份握手，其他跨 micro-batch Replay 保留原协议。继续扩展到 SP 或包含 TP collective 的 Replay 需要独立证明全 rank 同序发起。

性能结论来自 Seq2048/MBS1 和单节点 A100。每个 NSYS 用例只捕获一个预热后的训练步，无法推断长期的并发命中率。仅对 S2-DP/S3 做了三轮配对。4B 的 S0/S1/S2 各只有一次 12 步短跑。未采集 GPU 硬件资源计数器，也未测试长序列或包含 TP collective 的 Replay。

建议保留固定桶原型供审阅，优先研究在主 Stream 利用 NCCL 自带并发的调度方式。可以在更接近目标层 Backward 的已就绪桶提交后触发，再衡量缩短缓存驻留对显存和整步耗时的影响。推广独立 Stream 前，应以可重复的收益和 TE workspace 安全为门槛。

## 7. 附录：数据溯源与档案索引

| 证据 | 文件 |
|---|---|
| 源码版本 | 提交 `53c3660bab6a2396f061e5f5a6d7d80555a2cd9a`，`source_53c3660ba.tar.gz`，初始工作树 `git_status.txt`，PoC 的源码改动 `poc_source.patch` |
| 基线重复 | `perf_mb8_r{1,2,3}/S*/` 的 `manifest.json`、`train.log` 和 `rank*.json`，`perf_summary.json` 内含每轮每 rank 原始记录和配对差 |
| 通信时间线 | `small_s2/`、`4b_s2/`、`sp_control/` 的完整 `trace.nsys-rep`、`trace.sqlite`、`trace_analysis.json`、`tasks.json` 和 `comm.json`。采集诊断在 `nsys_diagnostics.json`。SP 配置拒绝见 `sp_rejection/train.log` |
| 命令与软件栈 | `run.py`、`capture.py`、`poc.py`、`profile_entry.py`、`poc_entry.py`、各用例 `manifest.json`，`nsys_version.txt` 和 `gpu_environment.txt` |
| 初期失败 | `perf_mb8_r2_failed/` 保存脚本名遮蔽 Python 标准模块的启动失败。PoC 首批和诊断运行保留首步桶初始化及 Saved Tensor 钩子导致的 DDP 梯度就绪计数断言 |
| 全量正确性 | `4b_comparison.json`、`small_fenced_comparison.json`、各验证用例的 `exact_rank*.json`，以及 `p9_correctness_reference/` 内的历史逐 rank 对照 |
| 并发原型 | `poc_4b_{same,side}_trace_conn1_fenced/` 的 NSYS 文件和逐 rank 数据，`poc_summary.json` 的 `timeline` 字段。示意图 `kernel_overlap_rank0.png` 由 `plot.py` 从原始区间生成 |
| 端到端测量 | `poc_4b_*_perf_conn1_*/train.log`、`rank*.json`、`manifest.json` 和 `poc_summary.json` 的 `perf` 与 `paired_perf` 字段 |
| 源码校验 | `source_sha256.txt` 记录基准源码包、PoC 补丁及实验脚本的校验值。 |
