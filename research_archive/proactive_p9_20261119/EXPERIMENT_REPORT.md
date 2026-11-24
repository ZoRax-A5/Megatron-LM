# P9 · 跨 micro-batch 主动重计算的正式训练验证

## 1. 核心摘要（Executive Summary）

**P9 PASS。** 正式 Megatron PP1 训练入口已接入默认关闭的双 micro-batch 窗口。真实 BF16/TE QKV 在第二个 micro-batch 的 Forward 中主动执行第一个 micro-batch 的 Replay，后续 Backward 消费缓存。工程上可以在测试配置中继续使用该实验开关。

- 单卡 2/4/8 个 micro-batch、TP2/DP1 和 TP2/DP4 的三步全量验证通过，S1、S2 的归约梯度分片和更新后参数对 S0 的最大绝对误差均为 0。
- TP2/DP4 的 30 步连续运行中，S2 共命中 1,440 个 checkpoint，释放全部 3,840 个，单 rank 同时驻留最多 2 个。
- 独立 GPU 时间线的 144 次成功 Replay 全部发生在后继 micro-batch 的 Forward 中。三组 30 步对照分离了双 Forward 调度成本和主动 Replay 的增量成本。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 设备 | 单节点 8× NVIDIA A100-SXM4 80GB，互联拓扑 GPU 对之间为 NV12 |
| 软件 | PyTorch 2.7.0a0+79aa17489c.nv25.04，CUDA 12.9，Transformer Engine 2.2.0+c55e425，NCCL 2.26.3+cuda12.9 |
| 模型 | 两层小型 Qwen3 配置，由 Megatron 正式 GPT 训练入口构建。BF16/TE、RMSNorm、QK LayerNorm、GQA、RoPE、SwiGLU。随机初始化，固定 seed 1234，mock tokens |
| 工作负载 | Seq256，MBS1，两层都启用 QKV 重计算。单卡为 2/4/8 micro-batch，TP2/DP1 为 4，TP2/DP4 为 8，后者 GBS32 |
| 梯度与优化 | AdamW，Distributed Optimizer，原生异步梯度 Reduce-Scatter，400,000 元素的 DDP bucket 设置。PP1、CP1、EP1，sequence parallel 关闭 |
| S0 | 原始 Forward/Backward 顺序，On-demand |
| S1 | 双 micro-batch 窗口，On-demand |
| S2 | 与 S1 相同的窗口。在第二个 Forward 中按层触发第一个 micro-batch 的 QKV Replay |

每个正确性用例先运行两次 S0，再依次运行 S1 和 S2。全部 rank 比较三步、每一步的完整归约后梯度分片和更新后的全部参数。通过条件为实测最大绝对误差等于零。训练时间来自独立运行的训练日志，显存峰值在各 rank 的每个训练步重置后采样。时间线在另一次运行启用 CUDA Event 和 NVTX。

## 3. 正确性与观测完整性验证

| 配置 | 每 rank micro-batch | 每组更新步数 | 各候选模式完整张量检查数 | 最大绝对误差 |
|---|---:|---:|---:|---:|
| TP1/DP1 | 2、4、8 | 每配置 3 | 每配置 6 | 0 |
| TP2/DP1 | 4 | 3 | 12 | 0 |
| TP2/DP4 | 8 | 3 | 48 | 0 |

每次检查覆盖该 rank 的全部归约后梯度分片（含 padding）或全部更新后参数。TP2/DP4 每 rank 每步覆盖 213,376 个梯度元素和 853,504 个参数元素。S0 重复运行先通过相同的全量检查。各模式日志中的 loss 序列一致，TP2/DP1 和 TP2/DP4 亦比较了逐 rank 未格式化的 loss 标量。既有主动 checkpoint 和 P8 回归测试共 19 项通过。

TP2/DP4 中各 rank 的 checkpoint ID、step、micro-batch、层和命中顺序一致。成功缓存的 A 每层执行 1 次原始 Forward、1 次主动 Replay、0 次 Backward Replay。逐 micro-batch 的 TE 调用计数与各 checkpoint 的命中、消费、释放事件互相印证。30 步 S2 的剩余 2,400 个实例均走按需重放，属于实验调度中未计划预取的实例。所有预取目标均命中。

三种模式每 rank 每步各发起 2 次原生异步 Reduce-Scatter，均在第 8 个 micro-batch 的 Backward。30 步内无跳步或 NaN，成功 step 结束时均通过缓存清理检查。

## 4. 性能评估与定量测量

TP2/DP4 下每模式独立运行 30 步。耗时统计第 6–30 步，给出同一次运行内 25 步的均值与样本标准差。显存列为整段运行内各 rank 峰值的范围。

| 模式 | 整步耗时 ms | 相对 S0 | peak allocated MiB | peak reserved MiB |
|---|---:|---:|---:|---:|
| S0 | 211.69 ± 6.05 | 基线 | 84.54–84.60 | 114–118 |
| S1 | 186.18 ± 6.94 | −12.05% | 87.87–87.92 | 116–120 |
| S2 | 201.41 ± 6.54 | −4.86% | 88.38–88.42 | 118–118 |

按全部 rank 的最大值计算，S1–S0 增加 allocated 3.317 MiB（3.92%），reserved 增加 2 MiB（1.69%）。S2–S1 再增加 allocated 0.502 MiB（0.57%），reserved 最大值降低 2 MiB（1.67%）。S2 比 S1 平均多 15.23 ms（8.18%）。预热后的第 6–10 步与最后 5 步相比，各 rank 的 allocated 峰值保持一致。

S2 共 1,440 次缓存命中，占全部 3,840 个 checkpoint 的 37.5%。从预取提交至释放的 CPU 单调时钟跨度中位数为 15.64 ms，均值 18.57 ± 24.27 ms，范围 7.85–281.29 ms。

## 5. 深度分析与机制探讨

双 Forward 窗口会让第一个 micro-batch 的计算图在第二个 Forward 期间继续存活。S1–S0 测量这个调度改变的显存成本。S2 另外将 QKV 的 Replay 计算图保留到第一个 Backward，S2–S1 给出引入主动 Replay 后的整步峰值增量。两个模式的峰值可能出现在不同阶段，不能将峰值差直接解释为某个 checkpoint 的大小。

S2 的匹配条件包含训练 step、前驱 micro-batch、层和 QKV 模块。DP4 的最后一个 micro-batch 独立执行，依然是唯一允许梯度同步的 Backward。FIFO 窗口内的梯度累加和损失记录顺序得以保留。这些约束解释了为何可以更改 Forward 的相对时机，同时得到相同的累积梯度。

**GPU 顺序实测。** 在独立时间线运行的 8 rank、3 步中，全部 144 次命中均满足前驱 Forward 结束、后继 Forward 开始、实际 Replay 起止、后继 Forward 结束、前驱 Backward 开始、缓存消费与释放的顺序。实际 Replay 完成到消费的 CUDA Event 距离中位数为 15.45 ms，均值 36.86 ± 64.66 ms，范围 6.47–290.61 ms，包含启动阶段。

以 rank 0 的第 2 步第一个配对为例，A 的 Forward 在 14.96 ms 结束，B 的 Forward 在 15.00 ms 开始，A 第一层的 Replay 位于 16.88–18.15 ms，B 的 Forward 在 29.59 ms 结束。A 的 Backward 在 29.62 ms 开始，42.28 ms 消费该图，43.61 ms 记录释放。该 GPU 时间轴与分别记录的 CPU 调用顺序一致。

整步耗时同时包含 Python 调度、TP 身份握手和 checkpoint 记账。单次短测观察到的 S1–S0 时间差不能单独归因于某项机制。

## 6. 局限性、决策建议与后续工作

实验覆盖同一 CUDA stream、小型随机初始化 Qwen3 模型和合成数据，结论适用于测试所用的 BF16/TE 配置。时间对照每种模式只有一个 30 步序列，步间波动无法估计不同运行之间的变异。当前时间线使用 CUDA Event 定位同一 stream 上的提交边界，NVTX 标注 CPU 调用范围。测量未覆盖逐 kernel 的执行区间和通信并发。RNG 测试核对 Replay 前后的 CPU、CUDA 和 Megatron tracker 状态，模型运行关闭了 Dropout。

本轮结果支持在相同配置下审阅和继续实验。扩展到独立 CUDA stream 前，应验证 TE workspace 与再计算图的跨流生命周期，并检查 TP collective 与梯度通信的事件依赖。较大模型、长序列和其它并行拓扑需要另外验证。

## 7. 附录：数据溯源与档案索引

| 证据 | 档案 |
|---|---|
| 源码基线 | 提交 `0765127eb4f2273b9aada48a24f285abec53b843`，`baseline.tar.gz`，原工作树 `status.txt`、`preexisting.patch` 和 `untracked.tar.gz` |
| 当前工作树 | `final.patch` 和 `final_status.txt`。源码归档校验值见 `source_sha256.txt`。P9 接入在 `megatron/core/pipeline_parallel/schedules.py`、`megatron/core/tensor_parallel/random.py`、`megatron/core/transformer/attention.py` 和 `megatron/core/transformer/transformer_config.py` |
| 运行入口 | `run.py` 依次运行 `single --tag _r1`、`tp2`、`dp4`、`perf`、`timeline`。每组的 `manifest.json` 保存完整命令行和实验环境变量，`entry.py` 添加逐 rank 观察 |
| 正确性 | 各用例 `verification_rank*.json` 的 `checks`、`bucket_layout`，S0 保留全量张量参考。底层验证器沿用 `qwen3_bench/qwen4b_c/gradient_entry.py` |
| 生命周期和归约 | `rank*.json` 的 `steps[].checkpoints`、`calls`、`max_cached`、`sync`、`reduce_scatter` |
| 时间及峰值统计 | `train.log` 的 iteration elapsed time、`rank*.json` 的 `allocated`、`reserved` 和 `timeline`。后处理为 `summarize.py` 和 `summary.json` |
| 回归及失败记录 | `unit.log`，以及初次单卡启动时错误的验证器路径日志 `single.log` 和 `single_mb2/S0/train.log`。修复路径后的同配置输出位于 `single_mb2_r1/` |
| 设备 | `gpus.csv`、`topology.txt` 与训练日志中的 NCCL 版本 |
