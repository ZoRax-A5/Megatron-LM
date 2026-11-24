# P15 · TP 张量依赖与研究路线决策

## 1. 核心摘要（Executive Summary）

**依赖闭环验收 FAIL，当前 Qwen3-4B Dense 的 TP Replay PoC 为 NO-GO。** 本轮复核了最后一个 micro-batch 的一个 TP2 Backward All-Reduce，两个 rank 的组成员、调用序号、张量规格和调用栈相符，并将各自 NCCL Kernel 与跨 Stream Event 关联。源码确定了通信输入的计算方式，但未在张量级闭合结果的首个消费者、候选 Replay 的有效生命周期和可插入的执行边界，无法计算可靠的关键路径收益。

- **实际调用路径纠正：**样例来自 Megatron 并行线性层的 Backward；不能将其归为 TE 融合线性层。随后部分调用确实来自 TE。
- 两个 rank 各捕获到 76 个目标范围内的 All-Reduce；按序号、形状、dtype 与调用栈逐项比较均相符，但同成员不同 communicator 的歧义未被排除。
- 样例的通信 Kernel 单次驻留分别为 0.126 和 0.904 ms；这不是可隐藏的关键路径时间。候选第 1 层 QKV 原 Replay 历史中位数约 0.168 ms，提前图约 22 MiB/rank。
- 建议停止围绕当前 Dense TP All-Reduce 继续搜索 PoC 窗口；下一步优先做 PP2 的最小消息身份追踪，MoE EP 暂列后续探索。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 硬件 | 单机 8×A100-SXM4-80GB |
| 模型与调度 | Qwen3-4B，BF16/TE，TP2/DP4/PP1，原生 PP1；序列 2048，MBS 1，GBS 32 |
| 梯度通信 | Distributed Optimizer，异步梯度归约，40M bucket，`CUDA_DEVICE_MAX_CONNECTIONS=1` |
| 样例 | 第 3 个训练 step、最后 micro-batch 的第一个 TP 组 `[0,1]` Backward All-Reduce；候选第 1 层 QKV |
| 测量 | 一次 NSYS CUDA/NVTX 与 CUDA Event 诊断；记录 Python 侧 Tensor 身份、storage、版本、形状、步数与调用栈 |
| 对照与执行 | 沿用 P14 的原生 On-demand 配置；训练 3 步，采集第 3 步；插桩仅在独立诊断入口显式启用 |

张量记录使用存活 storage 包装对象的本地编号，并同时记录对象、offset、stride、版本及指针；编号仅在同一进程有效。Kernel 时间采用 GPU 时间戳；Host API 时间只用于提交顺序。没有开展无 profiler 的性能测量。

## 3. 正确性与观测完整性验证

| 检查 | 结果 | 覆盖范围 |
|---|---|---|
| 诊断训练退出 | 成功 | 8 rank、3 step；不构成完整梯度或参数验证 |
| 目标范围 All-Reduce 对齐 | 76/76 项一致 | rank 0/1；组成员、序号、形状、dtype、异步标志与 Python 调用栈 |
| 样例 Kernel/Event | 两 rank 均找到唯一记录与等待，候选 record 间各仅一个 Kernel | 仅选定调用；其他 Event 不作推广 |
| 输入与输出 storage | 各 rank API 前后记录到同一 Tensor 对象及 storage 编号 | 不证明 GPU 结果已完成或下游消费身份 |
| 完整数值比较 | 未测量 | 本次无新 Replay，不声称 P15 新增全量梯度验证 |

两个 rank 的 Kernel 名称均表明 BF16 sum All-Reduce；记录的组成员和序号支持两端语义对应。**没有捕获 communicator 的唯一身份**，因此仍不能严格排除成员相同但实例不同的通信组。Tensor 指针重复出现时本轮不会据此视为同一存储生命周期。

## 4. 性能评估与定量测量

| rank | Host API 时长（ms，单次） | NCCL GPU Kernel 驻留（ms，单次） | Event record / wait Stream |
|---:|---:|---:|---|
| 0 | 0.376 | 0.126 | 39 → 151 |
| 1 | 0.148 | 0.904 | 39 → 151 |

- 这是同一个诊断步骤中的单次观察，没有重复次数、统计区间或可信的相对训练基线百分比。
- P12 对第 1 层原位置 Replay 的 GPU Kernel 中位数为约 0.168 ms；P11 测得单图提前缓存增量约 22 MiB/rank。这些为历史不同实验条件下的数据，本轮未重新测量。
- 目标通信的 GPU 驻留、Stream 实际阻塞和整步关键路径暴露是不同量。后两者在本轮均**未测量**。

## 5. 深度分析与机制探讨

**源码级已知链路：**选定调用位于并行线性层的自定义 Autograd Backward。它先执行 `grad_output.matmul(weight)` 生成 `grad_input`，随后对该 Tensor 原位提交异步 All-Reduce；在等待结果之前还可执行权重梯度计算，最后等待并将 `grad_input` 返回给 Autograd。两端捕获到的 Python 调用栈与该路径一致。

GPU 侧每个 rank 的 NCCL Kernel 在通信 Stream 上提交，随后同 Stream 记录 Event，另一 Stream 提交对应 wait。记录和等待的 Host API 在 Kernel 实际执行前提交，说明 CPU 返回并不代表通信完成。该链路证明了局部 Stream 排序，**没有证明返回梯度的首个实际 GPU 消费节点**。

候选 QKV 的原始 Forward 输入已记录，其入口张量与 checkpoint 调用实参在源码中相连。这个事实并不足以证明 Backward 时张量仍有效、版本未变、Replay 所需 TE 状态安全，也不证明候选独立于尚未完成的通信结果。没有捕获完整 Replay 图、参数快照及该 Tensor 的消费链，不能证明合法地在 wait 前插入 Replay，也无法排除干扰关键生产者或 TP collective 次序。

通信输入后的权重梯度计算提供了**可能已有计算隐藏通信**的源码路径；本轮未把这段重叠和最终等待做完整 GPU 依赖归因，不能将其当作已测得的隐藏比例。P12 对原 Replay 的约 0.168 ms 只是原计算可被完全移除时的理想量级，不是本 TP 窗口的可实现收益上界。资源竞争也仅是待验证假设。

PP2 的已有资料显示不同 stage 与 micro-batch 具备独立执行结构，但消息和反向图身份尚未闭合，不能凭空闲区间证明可调度。MoE EP All-to-All 可能存在较大的数据交换与计算交错机会，但现有 Dense 资料没有测到其负载、关键路径或 Replay 依赖；两者的潜力目前只能定性比较。

## 6. 局限性、决策建议与后续工作

本轮仅一个 TP2 样例、一次有插桩的 profiler 运行；Python 调用序号不是独立的 NCCL communicator 标识。storage 的弱引用编号只追踪采集时存在的 Python storage 包装，不构成完整 alias、释放或地址复用证明。没有捕获逐 Tensor 消费者、TE 内部完整 Saved Tensors、候选 Replay 的版本随时间变化，也没有建立整步关键路径或实测等待上界。历史 22 MiB 和 Replay 成本不能替代该样例的增量实测。Profiler 和诊断插桩会改变时序，数据不支持端到端性能结论。

1. **TP Replay：NO-GO。** 不满足依赖闭环和收益门槛，不开发 PoC，也不再对当前 Dense TP All-Reduce 反复扩大搜索。
2. **优先 PP2 最小身份实验：**在调度器拥有真实 micro-batch 身份处记录 stage、消息序号、方向、shape 和 send/recv 完成依赖，验证两端匹配和目标 checkpoint 生命周期，再评估是否存在可移动工作。先做身份追踪，不改变调度语义。
3. **MoE EP：暂不排序为首选。** 只有获得真实 MoE 负载与 All-to-All 的依赖和暴露时间之后，才能比较其研究价值；不可把理论通信量直接视为可隐藏关键路径。

## 7. 附录：数据溯源与档案索引

| 内容 | 来源与字段 |
|---|---|
| 源码基线 | `e49a7201ff30a9e8a989f3df1afee62bdb25aedf`；原始训练源码未修改 |
| P14 与 P12 依据 | `../proactive_p14_20261124/EXPERIMENT_REPORT.md`、`sample_graph.json`；`../proactive_p12_20261120/EXPERIMENT_REPORT.md` |
| 新诊断配置 | `manifest.json` 的 `cmd`、`env`；`status.json` 的 `returncode`；`train.log` 为完整运行输出 |
| 跨 rank 原始记录 | `tensor_rank0.json`、`tensor_rank1.json`；`kind=all_reduce` 的 `sequence`、`peers`、`tensor_before/after` 和 `stack`；`kind=qkv_input` 记录候选入口 |
| 对齐计数 | `cross_rank_summary.json`；仅比较所列字段，未验证唯一 communicator |
| Event 与 GPU 数据 | `sample_graph.json` 的 `scope`、`kernel.gpu_start/end`、`event_candidates` 和 `unique_target_producer`；由 `analyze_sample.py` 生成 |
| 诊断脚本 | `poc.py`、`poc_entry.py`；复制至工作区相应实验目录运行，依赖原有 bench、权重与环境；不会在默认训练路径加载 |
| 真实轨迹与未上传资料 | 工作区 `qwen3_bench/results/proactive_p15_20261124/p15_4b_C0_trace_conn1_tp2/trace.sqlite`、`trace.nsys-rep`；权重与完整张量快照未上传 |
| 源码定位 | `megatron/core/tensor_parallel/layers.py` 的 `LinearWithGradAccumulationAndAsyncCommunication.backward`，以及 `megatron/core/transformer/attention.py` 的 `SelfAttention.get_query_key_value_tensors`；TE 路径依据环境安装源码，未修改 |
