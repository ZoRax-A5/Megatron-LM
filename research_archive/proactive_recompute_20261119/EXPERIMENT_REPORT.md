# 主动提前重计算原型验证报告

## 1. 核心摘要（Executive Summary）

本轮完成了现有 Megatron checkpoint 路径分析和一个独立的一次性主动重计算句柄原型。CUDA 单测显示，预执行图在 Backward 中被真实消费，Linear/MLP 输入梯度、参数梯度及一次 SGD 更新与 Megatron 按需 checkpoint 对齐；不调用预执行时仍回退到按需重放。**阶段一设计与单 GPU原型为 PASS；Megatron 层内接入、多 micro-batch/多 rank、Qwen3-4B训练、跨 step 生命周期与训练级显存验收尚未执行，因此整体研究任务为 INCOMPLETE，不能判定最终 PASS。**

- 原生 `CheckpointFunction` 在 `no_grad` 下运行 Forward，并在 backward 中才重放；它不会自动保存可提前消费的内部图。
- 原型只对显式创建的一次性 `ProactiveCheckpoint` 句柄有效，测试确认预执行总共只重放一次，Backward 未再调用重放函数。
- 测试覆盖执行成功命中与未预执行回退；暂未覆盖预执行异常、输入改写、随机算子、并发实例或分布式路径。
- 工程决策：**GO** 进入单层边界适配前的进一步正确性验证；**NO-GO** 将此原型用于正式训练或通信窗口调度。

## 2. 实验环境与方法

| 项目 | 配置/记录 |
|---|---|
| 源码工作树 | `Megatron-LM-layerwise`，分支 `research/layerwise-fine-recompute` |
| 基线提交 | `1e6e162a2c8becda003164b5395ae9b01c229b09` |
| 设备 | 单 CUDA 设备；具体 GPU 型号未记录 |
| 软件 | PyTorch `2.7.0a0+79aa17489c.nv25.04`；pytest CUDA测试 |
| 模块 | Linear(8→16) + GELU + Linear(16→4)；CUDA张量 |
| 对照 | Megatron `tensor_parallel.checkpoint` 按需重放；相同初始化权重、输入、随机种子 |
| 计时口径 | 未测量性能/吞吐；记录句柄事件及预执行前后 allocated/reserved 当前值与 CUDA 全局峰值计数 |
| 显存口径 | 输出缓存字节数与 Autograd Saved Tensor 的去重 storage 字节数分开记录；不是训练级总显存归因 |

原型接口为 `handle = ProactiveCheckpoint(fn)`、`output = handle(*inputs)`、在 backward 前调用
`handle.prefetch()`。关闭该 API并继续调用原 `checkpoint` 时路径不变；它没有修改既有调用点或
训练配置默认值。

## 3. 正确性与观测完整性验证

| 用例 | 覆盖范围 | 结果 |
|---|---|---|
| 预执行后 Backward | 输入梯度、模块参数梯度、SGD更新后参数；另检查重放调用次数与 cache-hit/consume/release事件 | 通过 |
| 未预执行回退 | 原始无梯度 Forward 后发生一次按需重放，输入梯度为解析值 | 通过 |
| 多实例交错消费 | 两个句柄分别预执行，反序 Backward，检查梯度和命中状态不串用 | 通过 |
| 测试次数 | 三个独立 pytest 用例 | 3/3通过 |

比较使用 `torch.testing.assert_close` 默认容差；未扩大到完整模型或所有参数分片。当前验证是完整
小型模块梯度/参数比较，不是随机抽样；多 rank 覆盖为未测量。

## 4. 性能评估与定量测量

本阶段目标是证明状态交接，不进行性能主张。运行耗时、重复测量的均值/标准差、整步吞吐和
真实训练内存差值均未测量。

| 指标 | Proactive原型 | On-demand对照 | 相对变化 |
|---|---:|---:|---:|
| 额外预执行重放次数 | 1次/实例（测试计数） | 0次前置重放 | 不适用 |
| Backward重复重放 | 0次（测试计数） | 1次按需重放 | −100%重放次数，仅适用于此测试函数 |
| 输出缓存字节 | 48 B（单次演示） | 0个提前缓存 | 不作性能比较 |
| Saved Tensor去重storage字节 | 1,296 B（单次演示） | 0个提前图 | 不作性能比较 |
| allocated当前值 | 8,522,752 → 8,524,288 B（预执行前→后） | 未测量 | +1,536 B；单次设备当前值，不代表单实例完整开销 |
| reserved当前值 | 23,068,672 → 23,068,672 B（预执行前→后） | 未测量 | 变化 0 B；单次观测 |
| CUDA峰值allocated/reserved | 8,524,288 / 23,068,672 B | 未测量 | allocator全局峰值，不归因于该缓存 |
| 缓存存活时间 | 0.0575 s（单次演示） | 0 | 不作性能比较 |

- 观测到的关键现象是执行责任由 backward 内按需重放转移到显式 `prefetch()`，且图被消费一次。
- 当前测试不能说明提前执行是否降低训练时间，也不能把 CUDA全局峰值变化归因为单个缓存。

## 5. 深度分析与机制探讨

**直接实测结果：**两个 CUDA用例通过。成功路径在原始 Forward 和主动重放后函数调用数为两次；
Backward 后调用数仍为两次，并触发 cache-hit、cache-consumed 和 release 事件。输入梯度、参数
梯度和 SGD 更新参数与按需 checkpoint 对照一致。未预执行路径发生一次按需重放且得到正确输入梯度。

**机制解释：**当前 `CheckpointFunction` 的原始 Forward 置于 `no_grad`，所以当时没有内部
Autograd图可保留。原型在显式触发时用 detached 输入和 `enable_grad` 新建图；自定义 Function
Backward 将上游梯度交给该图反传，再把重放输入梯度作为 Function输入梯度返回。参数梯度由图中
参数叶节点正常累积。该机制可以在 backward 正式运行前构造图，但图和 Saved Tensors 必须保持
可达，因此会抵消一部分 checkpoint 显存节省。

待验证假设包括 TE融合算子保存状态是否能跨提前执行/Backward安全存活、参数或输入的版本变更
如何与缓存失效配合、Megatron并行路径中的 RNG/collective顺序，以及多个 micro-batch 的缓存
峰值能否由预算限制。上述均非本轮实测结论。

## 6. 局限性、决策建议与后续工作

- 支持范围限于单 CUDA设备、单次句柄调用、tensor-only输入/输出、普通 PyTorch无随机性模块和同一默认 stream。
- 当前 `prefetch()` 同步执行；无独立 CUDA Stream、后台任务队列、动态调度或缓存预算淘汰机制。
- RNG状态会快照和恢复，但随机模块没有验证；TE FP8/FP4、分布式 saved activation、CUDA Graph、activation offload、collective均不支持。
- 预执行异常与输入版本失效回退代码尚无测试；多个 checkpoint实例、多 micro-batch和跨训练step清理尚未验证。
- allocated/reserved记录的是设备当前值和 CUDA allocator 全局峰值，不是此实例精确分配归因；Saved Tensor统计按 storage 地址去重，仍需和实际峰值对照。
- 未完成单层 QKV/FC1、TP2/DP4全部 rank、Qwen3-4B有限训练；未验证训练长跑显存稳定性。

后续优先级：

1. 补充失败回退、输入版本变化、随机性和多句柄/多 micro-batch生命周期单测，并量化缓存峰值与存活时间。
2. 将 API 限定接入一个确定性单层 QKV或FC1边界，比较 no-recompute、on-demand、proactive完整梯度及参数更新。
3. 扩展 TP2/DP4各 rank验证，并检查一致 collective顺序、训练step隔离和长跑显存。
4. 通过前述检查后再做 Qwen3-4B有限训练。独立 stream接入需使预取 stream等待输入生产事件、消费 stream等待重放完成事件；输入/参数更新必须等待缓存消费。通信调度仅能建立在这些依赖上，不能改变 collective顺序。

## 7. 附录：数据溯源与档案索引

| 证据 | 路径/字段 |
|---|---|
| 设计分析与实现协议 | `Megatron-LM-layerwise/docs/research/proactive_recompute_design.md` |
| 主动句柄与Autograd桥接 | `Megatron-LM-layerwise/megatron/core/tensor_parallel/random.py`：`ProactiveCheckpoint`、`ProactiveCheckpointFunction` |
| CUDA正确性测试 | `Megatron-LM-layerwise/tests/unit_tests/tensor_parallel/test_proactive_checkpoint.py` |
| 原始测试输出 | `prototype_pytest.log` |
| 单次生命周期与显存演示 | `prototype_events.json` |
| 生命周期事件字段 | `checkpoint_id`、`event`、`output_bytes`、`saved_tensor_bytes`、`cache_lifetime_seconds`、`memory`；由 `handle.events` 和 `handle.write_events(path)` 提供 |
| 基线提交 | `1e6e162a2c8becda003164b5395ae9b01c229b09` |
| 开始时源码快照 | `/tmp/recom-layerwise-before-proactive.patch`、`/tmp/recom-layerwise-index-before-proactive.patch`、`/tmp/recom-layerwise-untracked-before-proactive.tar.gz`；SHA256记录见设计文档 |
| 未执行档案 | 多卡训练trace、Qwen3-4B日志、all-rank梯度/参数报告、完整显存曲线：未生成/未测量 |
