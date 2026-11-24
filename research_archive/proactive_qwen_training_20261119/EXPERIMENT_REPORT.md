# Megatron 主动重计算真实训练接入与分布式正确性验证

## 1. 核心摘要（Executive Summary）

在 Megatron 的 TE BF16 QKV 路径上，主动 Replay 已进入真实 Qwen3-4B 训练：在对应 Backward 前触发计算图重建，并由 Backward 消费该图。3 步完整模型验证中，8 个 rank 的有效梯度分片与更新后参数对 On-demand 比较均为零差异；100 步压力短训中每个 rank 完成 800 次提前 Replay、命中、消费和释放，未发生回退、未消费缓存、跳步或 NaN。**对本文覆盖的 TP2/DP4、PP1、BF16、QKV、无序列并行路径，功能正确性通过；当前实现可进入审阅，但不建议直接开启长序列或其他并行/算子配置。**

- TE 融合 LayerNorm/ColumnParallelLinear 的计算图和 Saved Tensors 可以跨 Prefetch 与 Backward 保存、复用；On-demand 路径仍保留，主动模式默认关闭。
- TP2 下采用全 TP 组身份握手，确认 step/micro-batch/checkpoint 顺序一致后才 Replay；真实 Qwen 验证覆盖 8 rank 与 Megatron Distributed Optimizer 的有效归约梯度分片。
- 有界缓存默认最多 1 个未消费 checkpoint/rank。100 步每次缓存输出为 12 MiB；缓存平均存活约 75.5–79.2 ms，整次运行的峰值 allocated 为 22.30–22.32 GiB/rank。
- 本轮性能数据来自一次、每模式 3 步的短测，只作观测记录，不构成训练加速结论；本轮未执行 Seq8192 的 100 步验证。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 硬件 | 8×NVIDIA A100 80GB；TP/DP 验证使用同一节点 8 GPU |
| 软件 | Python 系统环境；PyTorch `2.7.0a0+79aa17489c.nv25.04`；Transformer Engine `2.2.0+c55e425` |
| 模型与权重 | Qwen3-4B Dense；使用历史转换权重；mock tokens |
| 数据并行配置 | TP2/DP4/PP1，BF16，Distributed Optimizer，原生梯度 Reduce-Scatter；sequence parallel 关闭 |
| 短程完整模型 | Seq2048、MBS1、GBS32；3 优化器 step；只对 layer 0 的 `attn_qkv` 选择性重计算 |
| 稳定性训练 | 同一配置和触发策略，100 step，无 profiler |
| 真实 TE 单层 | BF16 SelfAttention QKV，确定性输入；No-recompute、On-demand、Proactive 共用同一初始权重、输入及 loss |
| 缓存上限 | `proactive_recompute_max_cached_checkpoints=1`；主动功能通过 `--proactive-recompute` 显式启用，默认关闭 |

主动触发由 PP1 训练 schedule 在每个 micro-batch 的模型 Forward 与其 Backward 之间执行。该点所有 checkpoint 输入已就绪，且尚未开始本 micro-batch 的 Backward。当前提供 step、micro-batch、layer、module、TP size 和 rank 标识；TP 组先做一致性握手，再同序 Replay。记录 CUDA Event 时间戳与 NVTX 区间；初期使用当前 stream，不主张 GPU 并发。

整步耗时取训练日志的 iteration elapsed time；显存报告区分 CUDA allocated 与 reserved。全模型三模式运行各只有一次、3 个连续 step，表中给出 step 均值及样本标准差，受启动和短样本波动影响。已验证的主动触发点是 A（Forward 后、该 micro-batch Backward 前）；尚未实现/验证 B（其它 micro-batch 运行期间主动触发已就绪实例）。正确性由所有 rank 的归约梯度分片及参数快照比对，并辅以逐 rank机制事件日志。

## 3. 正确性与观测完整性验证

| 验证 | 覆盖与结果 |
|---|---|
| 原型异常与生命周期 | 单 GPU 18 项通过：未 Prefetch 回退、Prefetch 异常策略、输入/参数版本变化、重复 Prefetch/消费、Backward 异常、Dropout RNG、多输出部分 loss、冻结输入、别名/共享参数、释放及跨 step 清理 |
| 实际 TE QKV | No/On-demand/Proactive 输出、输入梯度、全部模块参数有效梯度和一次 SGD 更新后参数，逐项 `rtol=0, atol=0` 一致；TE 融合路径未替换为普通 Linear |
| TP2/DP1 | 两 rank BF16 TE QKV 测试通过；TP 组顺序握手生效 |
| TP2/DP4 | 八 rank、三 step TE QKV 模块测试通过；比较 Reduce-Scatter 后 DP-owned 梯度分片与更新参数 |
| Qwen3-4B 完整训练 | 8 rank × 3 step；On-demand 和 Proactive 各 48 个检查（每 step/rank 的归约梯度与参数）；坏元素 0，最大绝对差 0，最大相对 L2 差 0；每项张量覆盖 502,873,984 至 2,011,495,936 个元素 |
| Qwen3-4B 稳定性 | 8 rank × 100 step；每 rank 800 个 checkpoint，全部进入 consumed/released；无回退、跳步或 NaN |

完整模型采用历史验证程序的基线重复建立与比较流程。报告中的零差异是已保存全量张量检查结果，不是随机抽样；损失在三模式中逐 step 相同。单层测试也显式重复运行 No-recompute，确认确定性小型测试基线逐位一致。另对全模型 No-recompute 基线进行独立三步重复运行：全部 48 个 rank/step 梯度和参数快照文件与首轮逐文件 SHA256 完全一致。全模型正确性结果只覆盖这里记录的输入、权重、seed 和训练配置。

## 4. 性能评估与定量测量

### 三模式短程训练

| 模式 | iteration elapsed（ms，n=3 step，均值 ± 样本标准差） | 相对 No-recompute | 3-step Loss |
|---|---:|---:|---|
| No-recompute | 17,703.8 ± 4,892.9 | 基线 | 9.929667 / 9.817410 / 11.736780 |
| On-demand QKV | 38,429.6 ± 3,795.1 | +117.1% | 9.929667 / 9.817410 / 11.736780 |
| Proactive QKV | 35,631.3 ± 3,725.6 | +101.3% | 9.929667 / 9.817410 / 11.736780 |

- 该短测中 Proactive 的均值低于 On-demand，但每模式仅一次运行，step 之间波动大；不能据此认定主动调度提升吞吐。
- No-recompute 与两种重计算的 loss 序列完全一致；有效梯度及参数快照比较也完全一致。

### 100 步缓存与显存观测

| 指标 | 观测 |
|---|---:|
| 每 rank checkpoint 数 | 800（100 step × 8 micro-batch × 1 QKV layer） |
| 每 rank 生命周期事件 | forward_complete / prefetch_start / prefetch_complete / cache_hit / cache_consumed / released 各 800；fallback 0 |
| 提前输出最大缓存 | 12,582,912 bytes（12 MiB）/ checkpoint |
| 缓存存活时间 | 各 rank 平均 75.5–79.2 ms；观测最大值 220.9–334.3 ms |
| CUDA peak allocated | 22.302–22.321 GiB/rank |
| CUDA peak reserved | 23.162–23.898 GiB/rank |
| 稳定性 | 100/100 step；跳步 0、NaN 0；step 21–100 平均耗时 1,636.3 ms |

Replay 图中 Saved Tensor backing storage 的去重估算上限为约 4.057 GB；这是图引用到的底层存储估算，可能包含共享/别名存储，**不等于缓存新增分配**。缓存期间 allocator 观测的增量才反映实际额外驻留；单次三步记录中 Prefetch 前后 allocated 增量约 22–23 MiB/实例，reserved 无变化。100 步测量的总峰值包含模型、优化器和训练激活，不能单独归因于该实例。

## 5. 深度分析与机制探讨

**直接观测：** TE QKV 使用融合的 LayerNorm/ColumnParallelLinear。Prefetch 在 Autograd 启用时重新执行该融合模块，将其输出及动态图保存在 checkpoint 句柄；Backward 消费缓存输出并沿已创建的图反向传播，故不再调用第二次 QKV Forward。参数 `.main_grad` 梯度累积与实训 Reduce-Scatter 结果均被纳入比较。

**直接观测：** 每个 checkpoint 身份包含 step、micro-batch、layer、submodule 与 rank。缓存限制在每 rank 一个；训练 schedule 在 micro-batch Backward 前触发并等待当前 stream 顺序完成，随后 Backward 命中并释放。TP ranks 在重放前校验身份/顺序，以防 rank 局部输入就绪导致通信次序分歧。被测 QKV 在关闭 sequence parallel 时不触发跨 rank QKV collective。

**机制解释与边界：** 提前重建图会把图节点引用的 Saved Tensors 延长至 Backward，因此显存节省取决于重建计算状态与提前释放原 checkpoint 激活之间的权衡。当前缓存预算以实例数限制而非字节精确配额；12 MiB 输出不代表完整图状态的全部物理占用。实际 allocator 增量和运行 peak 已记录，但不能将单个实例增量简单线性外推到多个并发缓存。

**后续待验证假设：** 多个同时缓存的 checkpoint、长序列 QKV 的缓存峰值，以及独立 CUDA Stream 上 TE workspace / RNG / Saved Tensor 的跨流安全仍需单独验证。现有同 stream 结果不能证明这些执行条件安全。

## 6. 局限性、决策建议与后续工作

### 支持范围及限制

- 已验证 PP1、CP1、EP1、TP1/2、DP1/4、BF16 TE `attn_qkv`，其中真实 Qwen3-4B 覆盖 TP2/DP4；本轮未接 FC1。
- 当前主动 mode 要求 selective QKV checkpoint；配置默认关闭。缓存默认最多一个实例/rank，调度点由 PP1 schedule 固定在各 micro-batch 的 forward/backward boundary，尚无外部任意时机 API、查询界面或字节预算器。
- Sequence Parallel selective fine checkpoint 当前由 Megatron 配置校验拒绝；TP2 SP=true 的尝试因此被拒绝，不能把结果外推到 SP。
- FP8/FP4、CUDA Graph、CPU offload、分布式 saved activation、PP/CP/EP、多流并发均未支持或被配置限制。
- Prefetch 失败默认终止该实例，避免部分副作用后错误回退。发生异常且参数已有 `main_grad` 时必须终止训练，因为 `.grad` 回滚无法还原 Megatron fused 梯度缓冲区。
- 100 步训练为 Seq2048/MBS1，未执行用户建议的 Seq8192/MBS2 负载；没有 profiler 时间线采集，也没有独立 CUDA Stream 或通信窗口调度实验。
- 没有做 100 步 On-demand 对照；本轮能支持的是 Proactive 的 100 步状态稳定性，不支持与 On-demand 的 100 步性能/内存因果比较。

### 决策建议

**核心机制 GO：** 对已测的 BF16 TE QKV、TP2/DP4、PP1、sequence parallel 关闭路径，提前图复用与梯度正确性通过，可进入代码审阅和受控验证。**本轮整体验收仍为部分未通过：** P4 要求的触发位置 B（其它 micro-batch 运行期间外部触发）尚未实现/验证，因此在完成该项之前不能宣称全部阶段验收 PASS。暂不将机制推广至其它模块、SP、FP8、PP 或独立 Stream，也不宣称训练加速。

下一阶段按优先级建议：

1. 审阅缓存字节统计和失效策略，补上可配置的真实缓存字节预算与任意 checkpoint 查询/触发接口。
2. 在相同节点验证多个 micro-batch 缓存策略和 Seq8192/MBS2 压力下的峰值显存；增加长程 On-demand 对照。
3. 逐步加入 FC1 或其它 TE 模块，各自先做完整单层梯度与异常验证。
4. 设计独立 Stream 前，先让 TE workspace、RNG、输入依赖及 Saved Tensor 的 stream 归属显式化；使用 CUDA Event 依赖保证生产、消费和释放顺序，并在 TP 组统一触发协议后再考虑通信窗口。
5. 通信并发阶段需在每个 rank 统一任务序列，禁止由局部就绪状态触发 collective；并把 DistOpt `main_grad`、bucket Reduce-Scatter 和异常终止语义纳入逐 rank 验证。

## 7. 附录：数据溯源与档案索引

| 正文证据 | 文件/字段 |
|---|---|
| 源码基线与改动 | 基线提交 `0765127eb4f2273b9aada48a24f285abec53b843`；`source_baseline_0765127eb.tar.gz`；`source_changes.patch`；`worktree_status_final.txt`。patch SHA256：`b7246d85ee94778740ee81a4aabf3df7cbc895542dc7b99e979fd1338a7f044c` |
| Checkpoint 图缓存设计 | `Megatron-LM-layerwise/docs/research/proactive_recompute_design.md`；实现见 `megatron/core/tensor_parallel/random.py`、`megatron/core/transformer/attention.py`、`megatron/core/pipeline_parallel/schedules.py`、`megatron/core/transformer/transformer_config.py`、`megatron/training/training.py` |
| 单卡异常与 TE 测试 | `p0_unit_tests_final.log`；测试源 `tests/unit_tests/tensor_parallel/test_proactive_checkpoint.py`；最终 18 passed |
| TP2 与 TP2/DP4 测试 | `p5_tp2_final.log`、`p5_tp2_dp4_final.log`；测试源 `tests/unit_tests/tensor_parallel/test_proactive_checkpoint_distributed.py` |
| TP2 SP 限制尝试 | `p5_tp2_initial.log`：sequence parallel 下 selective fine checkpoint 配置校验拒绝 |
| 三模式全模型配置、loss、耗时、梯度/参数检查 | `qwen3b4_seq2048_mbs1/validation_summary.json`；模式训练日志位于各模式目录的 `train.log`；8 rank 检查位于 `verification_rank*.json`；基线复测见 `no_recompute_repeat/` |
| 每 rank 全量快照 | `qwen3b4_seq2048_mbs1/{no_recompute,on_demand_qkv,proactive_qkv}/parameters_step*_rank*.pt` 与 `reduced_gradients_step*_rank*.pt` |
| Prefetch/消费事件 | `qwen3b4_seq2048_mbs1/proactive_qkv/mechanism_rank*.json`；事件名、时间戳、身份字段、内存字段；完整 stdout/stderr 汇总在 `proactive_qkv/train.log` |
| 100 步状态、显存、缓存 | `qwen3b4_stability100_seq2048_mbs1/stability_summary.json`；`memory_rank_*.json`；`mechanism_rank*.json`；`train.log`；`manifest.json` |
| 运行环境 | 各实验目录 `manifest.json`、`run_status.json`；环境版本与设备见本文第 2 节，未记录项目不作推断 |
| 测试和训练失败记录 | `p5_tp2_initial.log` 记录 SP 配置拒绝；`p5_tp2_no_sp_retry.log` 记录 TP2 无 SP 通过；最终日志与各运行目录的 `run_status.json` 保留成功状态 |
