# P17 补充验收：数值可重复性、EP 等待与重计算成本

## 1. 核心摘要（Executive Summary）

**P17 仍为 PARTIAL，P18 暂为 NO-GO。** 确定性配置下，EP2 的无重计算自对照及 `moe_fc1` 对照，在首次更新的已捕获完整张量上逐字节一致。EP8 代表性样本证实通信流事件与后续计算流之间的依赖，但合法的提前 Replay 窗口及可减少的整步时间尚未证明；重计算额外步时也未完成可加和的因果分解。

- 固定每 rank 初始模型、优化器状态、随机状态和每个 micro-batch 输入后，普通配置的两次无重计算仍有数值波动；确定性配置的全部捕获张量 SHA-256 一致。
- EP8 充分预热的前向 token Dispatch、前向 Combine、反向 Combine 驻留中位数分别为 **1.035、0.832、1.743 ms**；每项含 576 个 rank/层/micro-batch 调用，不能视为独立通信服务时间。
- EP2 profiler 对照中，`moe_fc1` 的每 rank/步 GEMM 总驻留增加约 **2.18 ms**，CPU 计时步增加约 **22.47 ms**。checkpoint backward 范围含原有反向计算，不能把范围总时长当作新增开销。
- 约 0.43B 参数的放大模型中，无重计算为 **284.70 ± 5.95 ms/步**，`moe_fc1` 为 **303.21 ± 4.93 ms/步**；两者 allocated 峰值都约 4.91 GiB，未见可分辨的显存收益。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 硬件与软件 | 单机 8×A100 SXM4 80GB、NVLink；PyTorch 2.7、TE 2.2、NCCL 2.26.3、NSYS 2025.2.1、BF16；与原 P17 基线相同 |
| 小模型 | 6 层、hidden 512、expert FFN 2048、8 experts、Top-2、约 0.11B 参数；Seq2048、MBS1、GBS32 |
| 放大模型 | 6 层、hidden 1024、expert FFN 4096、16 heads、8 experts、Top-2、约 0.43B 参数；其余数据与并行策略相同 |
| 并行与路由 | TP1/PP1/CP1、EP2 或 EP8、Dense DP8；原生 All-to-All dispatcher、TEGroupedMLP、mock tokens；无容量截断与按容量 padding |
| 确定性自对照 | EP2 首次更新；Megatron deterministic mode、NCCL Ring、TE 禁止非确定性算法、CUBLAS workspace `:4096:8`；与普通性能配置分开 |
| 正常性能 | 放大模型每策略 3 次独立 30 步，前 10 步预热，后 20 步统计；峰值在模型及优化器构建后重置 |
| 通信诊断 | 小模型独立 NSYS，20 步预热后分析第 21–23 步；单独启用 CUDA Event trace 检查 EP8 代表样本。放大模型为 10 步预热后 3 步 |

每 rank 每步 4 个 micro-batch；全局每步 65536 token。正常性能计时不使用 NSYS；allocated 指活跃 CUDA 分配峰值，reserved 指分配器保留池峰值。profiler 和带完整张量复制的数值诊断均不用于正常吞吐。

## 3. 正确性与观测完整性验证

固定状态的物理含义是，任何后续差异都不能由不同的初始权重、数据样本或随机数状态直接解释。分别保存每个 rank 首次 forward 前的可训练模型参数、优化器状态、PyTorch/TE、Python 和 NumPy 随机状态，捕获全部 4 个 micro-batch 的输入、Router map、GPT 训练 forward 输出、loss、完整 `main_grad` 及更新后模型参数。

| EP2 首步对照 | 初始/输入/Router/forward | 梯度 | 更新后参数 |
|---|---|---|---|
| 普通设置：无重计算 vs 无重计算 | 捕获张量逐元素一致 | 336,923,712 个元素不同；0 个超过原容差 | 476,056 个元素不同；137,884 个超过原容差 |
| 普通设置：无重计算 vs `moe_fc1` | 捕获张量逐元素一致 | 340,922,952 个元素不同；0 个超过原容差 | 484,200 个元素不同；141,704 个超过原容差 |
| 确定性：无重计算 vs 无重计算 | 每类完整张量字节哈希一致 | 全部字节哈希一致 | 全部字节哈希一致 |
| 确定性：无重计算 vs `moe_fc1` | 每类完整张量字节哈希一致 | 全部字节哈希一致 | 全部字节哈希一致 |

每组覆盖全部 8 个 rank；每 rank 94 个可训练参数张量、58,233,856 个梯度元素和同数目的更新后参数元素。普通设置的梯度最大绝对差分别为 9.18e-6 和 8.67e-6，参数最大绝对差均为 0.000244140625。预设逐元素阈值沿用原报告的 `atol=1e-5, rtol=1e-2`，没有放宽；逐张量不通过名称及误差分箱保留在机器数据中。

捕获的 GPT forward 输出是带标签训练时的**逐 token loss 张量**，并非完整 logits；没有保存每层全部中间激活或更新后优化器内部状态。确定性选项同时改变 TE、NCCL 和 PyTorch/CUBLAS 设置，因此可以建立可重复协议，不能单独证明某一个组件是普通设置波动的来源。该验证不覆盖不同 EP 切分之间的参数身份比较或长期收敛。

## 4. 性能评估与定量测量

放大模型正常性能为三次独立运行均值 ± 运行间样本标准差，显存给出各运行的跨 rank 最大值范围。变化相对放大模型无重计算。

| 配置 | 步时 ms | 相对变化 | token/s | 峰值 allocated GiB | 峰值 reserved GiB |
|---|---:|---:|---:|---:|---:|
| 无重计算 | 284.70 ± 5.95 | 基线 | 230,259 ± 4,853 | 4.910–4.916 | 6.186–6.568 |
| `moe_fc1` | 303.21 ± 4.93 | +6.5% | 216,181 ± 3,509（−6.1%） | 4.910–4.914（最大值 −0.04%） | 6.242–6.631（最大值 +1.0%） |

6 次运行均完成 30 步，未报告跳步或 NaN。allocated 峰值范围重合；这个指标包含模型和其他激活，并不能证明该边界从未节约局部张量。

下表为独立 NSYS trace 的 GPU Kernel 驻留，每种通信在每个配置各 576 个调用，数字为中位数 [全范围]，单位 μs；EP2 是相对基线。小模型取第 21–23 步，放大模型取第 11–13 步。

| 通信候选 | 小模型 EP2 | 小模型 EP8 | EP8 相对 EP2 | 放大模型 EP2 |
|---|---:|---:|---:|---:|
| 前向 token Dispatch | 82.64 [26.18, 6815.04] | 1035.43 [35.46, 2956.54] | +1153% | 100.59 [33.86, 3384.55] |
| 前向 Combine | 82.02 [25.22, 3804.86] | 831.50 [47.46, 2489.99] | +914% | 93.50 [32.58, 4232.17] |
| 反向 Combine | 79.22 [26.27, 5690.09] | 1743.31 [40.93, 6307.37] | +2101% | 102.45 [34.34, 5774.89] |

通信调用的 rank、层、micro-batch、EP 组、输入 shape 和 split 均有逐调用记录；反向 split 根据原生 Autograd 的输入输出交换规则还原。统计含 rank 间等待、消息差异及 profiler 扰动；三个 trace 的训练步数和学习率调度不同，不能把该对照当作独立通信带宽基准。

## 5. 深度分析与机制探讨

### Stream 依赖与可用窗口

**直接观测：** EP8 的三类候选均使用 NCCL SendRecv，通信 Kernel 位于 stream 51；在检查的 rank 0、第 22 步、layer 3、micro-batch 1 样本中，目标通信发射后，通信流记录事件，stream 7 等待同一事件代次，之后提交下一非 NCCL Kernel。审计确认从目标发射到事件记录间该通信流没有插入其他 Kernel。

前向 Combine 样本中，后续 Kernel 的 CPU 提交发生在目标 NCCL 结束之前，其 GPU 启动发生在目标结束约 11.6 μs 后。前向 token Dispatch 和反向 Combine 样本也有同类事件链；这些是具体样本而非全调用的关键路径统计。下一 Kernel 的名字为通用 elementwise Kernel，未凭名称声称它一定是目标张量的首个消费者。

原生调用顺序要求 Dispatch 结果进入排序与 Expert 计算，Combine 结果进入恢复排列；EP8 稳态 3456 个已归因通信 Kernel 与同 GPU 非 NCCL Kernel 的时间交集为零。**事件顺序证明样本依赖，不提供可消除等待的准确时长。** Event trace 本身可能扰动事件与流，且未测量硬件资源占用。

PP1 的当前训练调度按 micro-batch 执行 forward 后立即执行对应 backward，因此不存在已完成 forward、尚未 backward 的其他 micro-batch；同一 micro-batch 较早层的 checkpoint 是理论候选。现有 MoE `CheckpointFunction` 只在 Autograd backward 中重放；尚未实证候选在目标通信期内的输入、RNG、权重版本与可微图生命周期均可供提前执行，也未测得真实关键路径收益。

### `moe_fc1` 的额外成本

独立 profiler 中，24 个 rank-step 观测的均值如下；这是每种配置各一次 trace、非独立重复，GPU 驻留和 CPU 区间可能重叠，不能相加。

| EP2 小模型，ms/rank/步 | 无重计算 | `moe_fc1` | 变化 |
|---|---:|---:|---:|
| CPU 训练步区间 | 305.39 | 327.86 | +22.47 |
| 全部 GEMM GPU 驻留和 | 18.86 | 21.04 | +2.18 |
| 全部 NCCL GPU 驻留和 | 96.48 | 102.88 | +6.41 |
| 其他 GPU Kernel 驻留和 | 29.50 | 29.50 | +0.00 |
| CUDA runtime CPU 区间和 | 75.47 | 77.08 | +1.61 |
| 其中名称含 Synchronize 的 CPU 区间和 | 31.03 | 31.59 | +0.56 |
| checkpoint backward CPU 包含区间 | 未设置 | 28.25 | 不可直接比较 |
| 其中 Replay forward CPU 包含区间 | 未设置 | 8.70 | 不可直接比较 |

Replay 每 rank/步有 24 次调用；CPU 区间覆盖 Python/Autograd 和设备发射等内容，checkpoint backward 还包含本来就需要的 FC1 backward。全 NCCL 驻留变化包含普通 DP 等通信及等待，不能认定为额外通信工作。**这些观测排除了“额外步时完全等于 Replay GEMM”的解释，但尚不能闭合 27.18 ms 的无 profiler 步时差额。** 需要逐阶段、无嵌套重复计数的 Autograd 与主机等待对照，及多次 profiler 复核。

只有 Replay 的已就绪计算和其必要主机发射，且不引入新的关键路径等待时，提前执行才有机会减少暴露开销；移动 GEMM 本身不保证消除 Autograd、同步或通信变化。

放大模型的单次诊断中，全 GEMM 驻留和从小模型每 rank/步 18.86 ms 增至 48.50 ms，全 NCCL 驻留和分别为 96.48 和 94.54 ms；两类求和比由约 5.12 降至 1.95。它包含所有 NCCL 操作和可能的重叠，且两次 trace 预热与训练步数不同，只说明所捕获 Kernel 构成发生变化，不是隔离通信/计算成本比。

## 6. 局限性、决策建议与后续工作

- **验收判断 PARTIAL。** 确定性 EP2 首步数值自对照和 `moe_fc1` 对照通过；普通配置参数容差仍失败。未验证 EP8 的完整梯度/参数、放大模型正确性、更新后优化器状态或长期稳定收敛。
- 仅对三个 EP8 调用样本审计 Event 链，未验证首个计算 Kernel 的张量身份、全部 rank 的端到端关键路径或合法的提前 Replay 生命周期。通信驻留、事件依赖、真实暴露等待和资源竞争是不同量。
- 稳态 NSYS 每配置仅一次，样本内 rank 和调用不独立；计时受 profiler 及事件追踪扰动。mock data、单机拓扑、固定序列和仅一个放大模型均限制外推。
- 优先补充逐张量或多层级 backward 时间线，以不重叠的主机发射、Autograd、同步和 GPU 区间解释步时差；对普通模式分别隔离 TE、NCCL 和确定性选项，定位数值波动。
- 再验证少数同 micro-batch 较早层 checkpoint 的保存输入、RNG 和权重版本在 EP8 通信期的合法可用性，并测量不干扰训练的真实暴露等待；同时复核实际 token 数据与更大工作负载。
- **P18 NO-GO。** 首选待验证组合是 EP8 前向 Combine 或 token Dispatch，与同 micro-batch 较早层的 expert FC1 Replay：该通信有 Event 依赖，Replay 有实际成本。但目前尚无已证明合法且足够长的窗口，也没有可靠的整步可归因收益。

## 7. 附录：数据溯源与档案索引

- 原始 P17 提交为 `56a7f491c6434eb67bc2f5bdfd3e2c8ff07e98fd`；默认 Megatron 训练源码未修改。`run_correctness.sh`、`run_trace.sh`、`run_event.sh`、`run_large.sh`、`run_large_trace.sh` 与对应显式入口保存完整命令。
- `correctness_entry.py` 捕获初始参数、优化器/RNG、输入、Router、forward 输出、loss、`main_grad` 和更新后参数；`compare.py` 产生 `self_compare.json`、`fc1_compare.json`、`deterministic_self.json`、`deterministic_fc1.json`，逐张量记录 `different`、`violations`、`max_abs` 与误差分箱。`byte_hash.py` 生成 `byte_hashes.json`，按 rank 和类别哈希完整张量原始字节及名称、shape、dtype。
- `trace_entry.py` 显式 NVTX 标注调用身份，未调整原生调度；`trace_analysis.py` 关联 NVTX、CUDA runtime correlation ID 与 NCCL Kernel。`ep{2,8}_calls.jsonl` 与 `large_ep2_calls.jsonl` 每行记录 `meta`、消息 splits、CPU/GPU 时间和 stream；反向 shape/splits 根据 forward 元数据按原生 backward 交换规则还原。
- `trace_ep2.json`、`trace_ep8.json`、`trace_large.json` 的每项 `n/median_us/min_us/max_us` 对应 576 次调用。`idle_ep{2,8}.json` 由 `idle_analysis.py` 计算每个通信 Kernel 与同进程非 NCCL Kernel 的时间交集。
- `event_sample.py` 和 `event_edges.py` 对应 `event_sample.json`、`event_edges.json`：event 与 generation 匹配记录和等待，`only_target_launch` 表示目标发射到记录间没有额外同流 Kernel；CPU 提交时间与 GPU 时间均为 NSYS 时间戳，不把 StreamWaitEvent API 耗时视为 GPU 实际等待。
- `cost_analysis.py` 对应 `cost_ep2.json`、`cost_ep2_fc1.json`、`cost_large.json`，按进程和步区间归集 GPU Kernel 驻留和及 CPU runtime/NVTX 包含区间；`large_performance.json` 对应 `summarize_large.py`。`logs/` 保存正常训练、trace 和数值运行日志及显存记录，包含首次 NVTX 格式标记失败的日志。
- NSYS 原始 `.nsys-rep`、导出的 `.sqlite`、完整 `initial.rank*.pt` 与 `observed.rank*.pt` 未上传，保存在 `/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_followup_20261125/`。完整张量含参数、梯度和输入，应按该路径访问；这些文件未提供长期训练检查点。
