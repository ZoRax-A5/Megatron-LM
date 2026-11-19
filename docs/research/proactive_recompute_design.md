# 主动提前重计算：技术设计与验证边界

## 当前机制与结论

`tensor_parallel.checkpoint` 调用 `CheckpointFunction.apply`。其 `forward` 在
`torch.no_grad()` 中运行目标函数，仅保存输入和 RNG 快照；输出连接到自定义
Autograd 节点。该节点的 `backward` 才恢复输入/RNG、以 detached 输入和
`torch.enable_grad()` 重放函数，并对重放输出调用 `torch.autograd.backward`。

**结论：计算图可以在自定义 checkpoint 的 backward 前创建并保留，但现有
CheckpointFunction 不会自动接收或复用它。** 图必须由相同重放函数、正确版本的
输入和参数创建；之后仍需在 checkpoint backward 中以实际上游梯度对图反传，并
把输入梯度返回给外层 Autograd。缓存输出值而不缓存其 `grad_fn` 图和保存张量不足以
完成反向传播。提前图会保留算子 Saved Tensors，因而会消耗被重计算原本释放的显存。

## 方案比较

| 方案 | 状态交接 | 优点 | 主要风险/成本 |
|---|---|---|---|
| A. 扩展 `CheckpointFunction` | 每个 autograd `ctx` 保存 checkpoint ID、输入/RNG元数据；前置 API 将重放图原子交给该 ctx；backward 消费图并反传 | 生命周期和唯一性天然绑定 Autograd 节点；不用全局猜测 layer 身份 | `ctx` 在调用者持有输出期间不可直接访问；需要可寻址句柄注册表；异常、图释放、并发和 reentrant backward 状态更复杂；需维护自定义 Function 与 TE checkpoint 的差异 |
| B. 独立 `ProactiveCheckpoint` 管理器 | checkpoint 包装对象持有唯一实例 ID、运行函数、保存输入/RNG和显式状态机；自定义 Function backward 通过对象消费预建图 | 可显式触发/统计/清理；状态转移和测试容易独立验证；不依赖 layer 命名 | 管理器必须严格绑定一次 forward 调用；不能按 module 或 layer 单例复用；持有输入/图会增加显存；需保证 backward 单次消费和失败回退 |

最小实现采用 B：一个对象只对应一次 checkpoint forward，作用域限定单设备、无随机性、无
collective的纯 PyTorch 函数。其稳定身份由单调递增序号生成；`prefetch()` 显式重放，
`CheckpointFunction.backward` 取走缓存图，施加真实上游梯度，收集 checkpoint 输入梯度，
并释放图引用。未提交、预执行失败或输入版本不匹配时走按需重放。成功缓存不能被第二次
消费；第二次 backward 明确报错。事件记录覆盖 forward、submit/start/complete、consume/hit、
fallback、release，并记录缓存字节与存活时间。

## 正确性协议

1. 正常 forward 在 `no_grad` 下执行并保存原始输入、输入版本、必要 RNG 快照。
2. `prefetch` 在同一 CUDA stream 上对输入建立可微 detached 副本，重放一次并保留输出图；
   通过 Autograd 图自然保存的参数引用和 Saved Tensors 保持存活。
3. checkpoint backward 对输出图调用 `torch.autograd.backward(outputs, grad_outputs)`，然后
   将重放输入的梯度作为自定义 Function 的输入梯度返回。参数梯度由该图的叶参数累积。
4. 前置执行若检测到 stale 输入、重放异常或结果不完整，应丢弃缓存并重新按需重放；不可
   把预执行输出直接当作 backward 的结果。缓存只能消费一次并在消费后清理。

RNG 重放需要在预执行时恢复原 forward 的 CPU/CUDA/Megatron tracker 状态，并在其后恢复
调用者当前 RNG 状态；TE FP8/FP4、CUDA Graph、activation offload、分布式激活切分以及含
collective的函数在最小原型中不支持。现有 `distribute_saved_activations` 会改写输入存储，
必须在启用主动模式前另行定义恢复与版本校验协议。TP/EP下提前执行还必须确保所有 rank
按相同 collective 顺序到达对应操作；本原型不触及该路径。

## 当前层级边界

仓库的层内细粒度边界由 Attention/MLP 的 `tensor_parallel.checkpoint` 调用定义，覆盖
例如 `attn_qkv`、`attn_proj`、`mlp_fc1`、`mlp_act`、`mlp_fc2`。上述 checkpoint 与
TransformerBlock 的按层 chunk checkpoint 共用同一套 reentrant Function。TE checkpoint
另由 `te_checkpoint` 实现，处理 FP8/FP4 元数据和量化上下文，不能假定可直接换用本原型。

## 后续接入步骤

先完成纯 PyTorch 单 GPU线性/MLP完整梯度与参数更新比较，再接单个 QKV/FC1 边界，然后
验证多实例/micro-batch状态映射。只有原型复用与释放证据通过后，才考虑 Megatron训练循环
触发接口、TP/DP所有 rank验证及 Qwen3-4B有限训练。独立 CUDA Stream接入时需增加事件依赖：
prefetch stream等待输入生产事件，消费 stream等待重放完成事件；参数更新/输入覆写之前必须
完成消费。通信窗口选择器应置于这些正确性协议之上，且不能改变 collective 顺序。

## 工作树基线

分析基于 `1e6e162a2c8becda003164b5395ae9b01c229b09`。开始时工作树已有 layerwise
重计算相关未提交改动及两个未跟踪文件；这些改动未被覆盖。开始时状态快照位于
`/tmp/recom-layerwise-before-proactive.patch`、
`/tmp/recom-layerwise-index-before-proactive.patch` 和
`/tmp/recom-layerwise-untracked-before-proactive.tar.gz`，SHA256分别为：

- `eecb62052fb1d23b84bfbbc7fa7793f1495d18920d875105e85506079dc2e1ac`
- `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `94d475eb91290d1e3aed12f191099704a0927fb07e40751ea53c8ba88b26c258`

## 2026-11-19 原型完善与本地 QKV 接入补充

P0增加了输入、参数和 buffer 的 PyTorch version-counter 快照，并把追踪到的参数作为自定义
Autograd Function 的显式输入，使输入 `requires_grad=False` 时仍可触发参数反向。调用者显式
传入的参数与从 callable、闭包和绑定模块发现的参数合并并按对象身份去重。输入别名仍按原始
Function参数返回梯度，由外层 Autograd 汇合；共享参数只追踪一次，由内部图按正常 Autograd
规则累积。Backward开始前同时读取 `ctx.saved_tensors`，启用 PyTorch 原生版本校验。

Prefetch异常默认进入 terminal failure；只有调用方明确声明 `allow_prefetch_fallback=True`
时，且版本检查确认输入、参数和 buffer 未变化，才进入按需回退。该声明表示调用方确认重放
函数没有无法检测的 Python/外部副作用。Backward异常不尝试重新计算；参数已有 `.grad` 会先做
快照，若内部反向抛错则恢复原 `.grad`，释放缓存并进入 `backward_failed`。`release()` 可显式
放弃未消费实例；registry 使用弱引用，成功消费/放弃后注销实例。支持 tensor 和 `None` 的
多输出，并支持随机算子：预执行期间恢复原Forward RNG，再恢复调用者当前 RNG。Dropout测试与
按需模式梯度一致。

P1仅接入 `SelfAttention.get_query_key_value_tensors` 中已有 `attn_qkv` 边界。新增
`proactive_recompute` 配置开关，默认关闭；启用时要求 selective、`attn_qkv`、local
`ColumnParallelLinear`、非FP8/FP4及非 offload/CUDA Graph路径。实例元数据包含本地层号和模块名，
调度调用方通过 `pending_recompute_checkpoints()` 获取可触发ID，调用
`prefetch_recompute(checkpoint_id)` 执行预取；`get_recompute_checkpoint(id)` 用于查看事件。
On-demand分支保持原调用不变。TE Linear/融合算子没有接入，FC1也没有接入。

本地QKV CUDA测试以相同初始权重、输入、平方损失和 SGD 比较 no-recompute（重复两次）、
On-demand和Proactive，验证输出、完整输入梯度、全部参数梯度和更新后参数。Proactive的
调用计数为原始Forward 1、前置Replay 1、Backward Replay 0；事件记录确认cache-hit、consume
及release。测试和显存快照见
`qwen3_bench/results/proactive_recompute_qkv_20261119/`。这些结果只覆盖单卡单层本地QKV，
不构成多 micro-batch、多 rank或端到端训练证明。
