# P17：MoE Expert Parallelism 基线与通信特征

## 1. 核心摘要（Executive Summary）

**P17 判定 PARTIAL（未达到 PASS）；P18 暂缓（NO-GO）。** 八卡真实 MoE EP2/4/8 已完成重复训练，识别到 dispatch 与 combine 的 GPU 通信，并测得现有 `moe_fc1` 的运行成本。当前尚未建立可通过逐元素参数容差的独立运行对照，也未量化可靠的关键路径暴露，因而不能据此启动主动调度或声称吞吐收益。

- EP2 的稳态训练为 **259.55 ± 3.61 ms/步**，三种 EP 配置均有三次独立运行；本负载为约 1.1 亿参数的 mock-data 模型。
- EP2 和 EP8 的 trace 显示 dispatch、combine 对应 NCCL SendRecv Kernel；短 trace 中与同 GPU 非 NCCL Kernel 的交集为零，但真实等待与可合法安排的 Replay 时间均未知。
- 路由有非均匀性：按层汇总的最大专家负载与平均负载之比，EP2 为 1.08–1.96，EP8 为 1.06–1.75；无容量截断配置且抽样的每个 token 有两次分配。
- EP2 的 `moe_fc1` 重计算增加约 **10.5% 步时**，未观察到明确的 allocated 峰值下降；参数严格比较失败，且无重计算重复对照也失败。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 硬件和互联 | 8× A100 SXM4 80GB；GPU 两两 NV12 |
| 软件 | PyTorch 2.7.0a0+79aa17489c.nv25.04；TE 2.2.0+c55e425；NCCL 2.26.3；NSYS 2025.2.1；BF16 |
| 模型 | GPT、6 层、hidden 512、attention heads 8、expert FFN 2048；8 experts、top-2；约 0.11B 逻辑参数 |
| 数据和优化 | mock data、NullTokenizer、词表 1024、seed 42、Adam；序列 2048、MBS 1、GBS 32、学习率 1e-4 至 1e-5 |
| 路由 | 原生 alltoall dispatcher、TEGroupedMLP、无线性 bias；aux-loss 系数 0，容量因子未设置，不按容量 padding |
| 并行 | TP1、PP1、CP1，EP2/4/8；Dense DP 均为 8，expert DP 分别为 4/2/1 |
| 测量 | 每配置 3 次独立 50 步，无 profiler；前 20 步 warmup，后 30 步逐步日志取均值。另做 3 步 NSYS 诊断和独立路由统计 |

每 rank 每步处理 4 个 micro-batch、8192 个输入 token；全局每步 65536 个 token。allocated 和 reserved 分别是 PyTorch CUDA 分配器活跃分配峰值与缓存池保留峰值；均在模型和优化器构建后重置，覆盖训练阶段。EP2 首次运行未启用峰值插桩，其显存表仅有后两次。

性能计时来自正常训练整步日志；吞吐以全局 token 数除以该运行稳态平均步时。NSYS 的 GPU Kernel 驻留是独立诊断，不用于计算正常吞吐。路由采集含 GPU 到 CPU 同步，仅用于统计分配，不用于性能结论。

## 3. 正确性与观测完整性验证

| 验证 | 覆盖与结果 |
|---|---|
| 原生训练 | EP2/4/8 各 3 次、每次 50 步；日志均未报跳步或 NaN，末步 loss 及梯度范数见原始日志。短程稳定不代表收敛等价。 |
| 类型和通信 | 运行时确认 `MoEAlltoAllTokenDispatcher` 和 `TEGroupedMLP`；调用链通过 `all_to_all_single`，GPU trace 归因到 `ncclDevKernel_SendRecv`。 |
| `moe_fc1` 数值 | 相同 EP2 配置、首步、所有 8 rank；每 rank 94 个可训练张量、58,233,856 元素的 `main_grad` 及更新后模型参数，逐元素比较，容差 `atol=1e-5, rtol=1e-2`。梯度 8/8 rank 通过；参数 0/8 通过。 |
| 独立运行对照 | 无重计算对无重计算，同一比较条件：梯度和参数均为 0/8 rank 通过；参数最大绝对差同为 0.000244140625。 |

梯度与参数均无非有限值，也没有缺失的 `main_grad`。重计算对照的梯度最大绝对差为 9.30e-6、最大相对 L2 为 0.00202；参数最大相对 L2 为 0.000185。对照显示独立运行已有同量级波动，但尚不能证明其原因或放宽验收容差。未进行跨 EP 参数逐元素比较，也未比较优化器内部状态。

## 4. 性能评估与定量测量

表中步时和吞吐均为三次运行的均值 ± 运行间样本标准差；显存为各运行跨 rank 最大值的范围。变化相对 EP2 无重计算基线；显存变化以跨运行峰值最大值计算。

| 配置 | 步时 ms | 步时变化 | token/s（变化） | allocated GiB（峰值变化） | reserved GiB（峰值变化） |
|---|---:|---:|---:|---:|---:|
| EP2，无重计算 | 259.55 ± 3.61 | 基线 | 252,533 ± 3,492（基线） | 1.512–1.512（2 次，基线） | 1.674–1.678（2 次，基线） |
| EP4，无重计算 | 270.26 ± 14.56 | +4.1% | 242,946 ± 12,703（−3.8%） | 1.176–1.183（−21.8%） | 1.709–1.711（+2.0%） |
| EP8，无重计算 | 256.06 ± 8.11 | −1.3% | 256,111 ± 8,005（+1.4%） | 1.003–1.006（−33.5%） | 1.500–1.527（−9.0%） |
| EP2，所有层 `moe_fc1` | 286.73 ± 0.74 | +10.5% | 228,566 ± 588（−9.5%） | 1.511–1.515（+0.2%） | 1.672–1.680（+0.1%） |

- EP 大小变化时端到端步时差异相对运行间波动并不稳定，不能单独归因于通信。
- 本配置下 `moe_fc1` 增加步时，峰值显存范围与基线重合；峰值包含其他分配，不能推出该边界完全没有节省激活。

独立 NSYS 中每个 rank、每步、每层执行一次 token dispatch、一次概率 dispatch 和一次 combine；下表为 8 rank、3 步、6 层所有匹配 Kernel 的中位数及全范围，单位微秒。每项各 576 次；范围含启动阶段，变化按中位数计算。

| 前向通信 | EP2 中位数 [范围] | EP8 中位数 [范围] | EP8 相对 EP2 |
|---|---:|---:|---:|
| Dispatch token | 83.39 [26.53, 3170.34] | 677.74 [34.98, 3509.73] | +713% |
| Dispatch 概率 | 11.62 [8.32, 850.11] | 120.96 [9.82, 667.49] | +941% |
| Combine | 80.24 [25.18, 93053.08] | 648.21 [37.60, 25968.03] | +708% |

反向 combine 与 dispatch Kernel 的 EP2 中位数分别为 83.10 和 64.69 μs，EP8 为 1807.52 和 339.81 μs；dispatch 反向合并了 token 与概率两类调用。相应 Kernel 样本数为 576 和 1152。诊断范围内 token dispatch 输入为 4096×512 BF16 元素，概率为 4096 个元素；combine 输入 token 数依负载而变。上述驻留受等待和 profiler 扰动影响，不是单独通信服务时间。

前向 Expert 区间内，EP2 匹配 4604 个 GEMM Kernel，中位数 19.62 μs；EP8 匹配 1152 个，中位数 53.28 μs。每个 router 区间的 GEMM 样本数为 576，两个配置中位数均为 6.62 μs。不同 expert 分片导致单次 Kernel 粒度不同，不能直接以单 Kernel 时间比较总计算量。

`moe_fc1` 单独 trace 观察到 576 次 Replay 调用和 2304 个匹配 GEMM Kernel，中位数 22.21 μs、总驻留 53.42 ms（8 rank、3 步合计，非整步增量）。Attention 与 FC1/FC2 的独立完整时间分解未测量。

## 5. 深度分析与机制探讨

**直接观测：** EP2 与 EP8 的 3456 个已归因 All-to-All Kernel 中，没有与同 GPU 非 NCCL Kernel 的时间交集。前向 dispatcher 先交换 token 和概率，再进入 expert 计算，之后 combine 返回输出；反向存在对应反向集合通信。

**尚待验证的解释：** EP8 驻留中位数较大可能与更多 rank 同步、专家分布或 Kernel 等待有关，但本实验没有硬件资源计数器、隔离通信服务时间或足以证明关键路径的事件链。驻留期间没有其他计算 Kernel 不等于 GPU 有可用计算容量，也不证明某个 Replay 已就绪。

Router 统计按层累加专家分配数后取最大值与平均值之比，用于描述负载倾斜；它不能代表真实语料。每层 EP2 与 EP8 各有 96 个 rank-micro-batch 观测，全部满足分配总数为输入 token 的两倍。单次观测的最大/平均比最高分别为 2.88 和 2.63，说明模型中存在不均匀性。

## 6. 局限性、决策建议与后续工作

- **验收未满足：** 严格参数数值协议未通过；可归因的等待、关键路径暴露、候选 Replay 就绪性、Attention/FC2 分解和跨负载路由变化未完成。不能将局部 trace 交集当作资源竞争证据。
- 仅研究单机 NVLink、单个小模型、seq2048 和 mock tokens；没有实语料、长时间收敛、通信隔离测试、硬件计数器或多节点结果。模型小于最初数亿参数建议，适合建立可控基线，不能外推更大模型。
- NSYS 每种配置仅一次三步诊断，含初始化和长尾；调用中位数并非独立运行的不确定度，也不可用来推导正常吞吐或潜在加速。路由统计仅覆盖前三步。
- 首要后续工作：固定相同初始状态和数据，增加无重计算重复的完整张量对照，定位独立运行波动；确定预先约定的 BF16 验证协议后重测 `moe_fc1` 与另一代表边界。
- 其次增加稳态、分阶段 trace 与消息 split 记录，确认 Expert GEMM/Attention 的分布、数据依赖和可证明的等待；再以真实数据或受控负载复核路由倾斜。
- **P18 NO-GO，候选而非已批准实验：** 如果上述验证确立合法的未隐藏窗口，可优先研究 EP8 token dispatch/combine 与 expert FC1 Replay 的资源竞争；原因是这组 trace 中通信驻留较长且已有真实 Replay 任务。现有证据无法判断其合法调度或端到端收益。

## 7. 附录：数据溯源与档案索引

- 源代码基线：`ea28d9c6b12545447d7f591d6c0bb1e6be89e1ba`；P17 工作树仅新增档案和显式调用的诊断包装器，Megatron 默认训练源码未修改。
- 命令：`smoke.sh`、`baseline.sh`、`diagnostics.sh`；`instrumented_entry.py` 显式启用时记录进程组、显存、路由或 NVTX。FC1 trace 使用与 `diagnostics.sh` 相同的 NSYS 命令并增加 `--recompute-granularity selective --recompute-modules moe_fc1`；独立基线数值对照使用 `P17_SNAPSHOT=1` 和 `--train-iters 2`。
- 预检失败与成功：`smoke.log`、`smoke_retry.log`、`smoke_retry2.log`。首个因 TEGroupedMLP bias/fusion 断言失败，第二个缺少 eval interval。
- 硬件/软件：`environment_nvidia.txt`、`environment_software.txt`、`topology.txt`；全部实际参数、训练 loss 与梯度范数见 `logs/*.log`。`logs/*.rank*.jsonl` 的 `groups` 给出每个 rank 的 TP、PP、CP、Dense DP、EP、expert DP 成员，`peak_allocated` 和 `peak_reserved` 单位字节。
- `performance.json` 对应 `summarize.py`：`stable_ms_mean/std` 为每次第 21–50 步，`tokens_per_s` 为 65536 除以该均值；显存字段是该运行所有 rank 的峰值最大值。EP2 的 `ep2_run1` 不含显存 JSON。
- `routing.json` 对应 `summarize_routing.py`，由 `logs/ep{2,8}_routing.rank*.jsonl` 的 `counts` 和 `tokens` 聚合。`max_to_mean_aggregated` 为每层累积专家分配的最大值除以平均值。
- `trace_ep{2,8}.json`、`overlap_ep{2,8}.json`、`compute_ep{2,8}.json`、`compute_fc1.json` 分别由 `analyze_trace.py`、`analyze_overlap.py`、`analyze_compute.py` 从 NSYS SQLite 的 NVTX、CUDA runtime correlation ID 和 GPU Kernel 关联生成；compute 分类按 Kernel 名称包含 gemm。`overlap_ns` 是每个已归因通信 Kernel 与同设备非 NCCL Kernel 时间并集的交集。
- `correctness.json`、`correctness_control.json` 由 `compare_snapshots.py` 比较本地 `.pt` 中全部同名张量；`allclose_atol_1e-5_rtol_1e-2` 为逐元素判断，`relative_l2` 为整体差值 L2 / 参考 L2。
- 未上传的大文件保留于 `/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_20261124/`：`*.nsys-rep`、`*.sqlite`、`ep2_snapshot_*.pt`。这些快照包含各 rank 首步完整 `main_grad` 和更新后模型参数；未保存模型权重训练检查点。原始日志和逐 rank JSONL 同时归档在本目录 `logs/`。
