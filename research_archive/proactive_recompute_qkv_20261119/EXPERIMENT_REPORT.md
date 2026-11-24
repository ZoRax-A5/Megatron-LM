# Megatron 主动重计算 P0–P2 验证报告

## 1. 核心摘要（Executive Summary）

本轮补强了 checkpoint 状态保护，并将主动机制接入 Megatron 本地实现的单层 `attn_qkv` 边界。**P0 原型保护、P1 QKV 接入及 P2 单 GPU正确性验证均 PASS：Proactive 的 QKV 原始 Forward、前置 Replay、Backward Replay 次数分别为 1、1、0；输出、完整输入梯度、参数梯度及 SGD 更新后参数与 On-demand 和 No-recompute 对照通过默认 `torch.testing.assert_close` 容差。**该结论仅适用于单 GPU、local `ColumnParallelLinear`、FP32、小型单层测试；TE、FC1、并行训练和真实模型尚未验证。

- 参数、输入和 buffer 使用 PyTorch version counter 检查；Prefetch异常默认终止，只有显式声明安全时才允许回退。
- 单 GPU原型测试 15 项通过；QKV模块三路对照和重复 No-recompute 基线通过。
- QKV缓存事件记录 1,920 B输出与 5,632 B去重 Saved Tensor storage；缓存消费后释放。
- 工程决策：**GO** 开始设计 TE/QKV兼容和多 micro-batch验证；暂不进入 TP2/DP4 或通信并发阶段。

## 2. 实验环境与方法

| 项目 | 配置 |
|---|---|
| 源码分支/基线 | `research/layerwise-fine-recompute` / `1e6e162a2c8becda003164b5395ae9b01c229b09` |
| GPU | NVIDIA A100-SXM4-80GB，单卡 |
| 软件 | PyTorch `2.7.0a0+79aa17489c.nv25.04`，CUDA测试 |
| 模块 | Megatron `SelfAttention.get_query_key_value_tensors` 的 local `ColumnParallelLinear` QKV投影；Seq=5、Batch=2、Hidden=16、FP32、单层 |
| QKV bias | 关闭，验证 checkpoint `(tensor, None)` 输出 |
| 三种路径 | No-recompute；原有 `tensor_parallel.checkpoint` On-demand；新 `proactive_recompute=True` |
| 主动触发 | `pending_recompute_checkpoints()` 发现实例ID，`prefetch_recompute(id)` 触发；`get_recompute_checkpoint(id)` 查询生命周期事件 |
| Loss/更新 | 三路相同输入和初始权重，`output.square().sum()`，SGD `lr=0.01` |

开关默认关闭。启用当前原型要求 selective recomputation 且选中 `attn_qkv`，并限制为 local
非 FP8/FP4、非 CUDA Graph、非 offload、TP/PP/CP/EP size均为1。TE QKV类在运行时明确报错。

## 3. 正确性与观测完整性验证

| 验证项 | 结果 |
|---|---|
| 原型保护及异常路径 | 15项单 GPU用例通过：未预执行回退、Prefetch异常终止/显式安全回退、输入及参数版本变化、重复 Prefetch/Backward、Backward异常参数梯度回滚、Dropout RNG、部分多输出、输入冻结、共享参数与输入别名、实例交错、显式释放、连续5次step生命周期 |
| QKV Forward/梯度/更新参数 | No-recompute重复2次；On-demand 1次；Proactive 1次；全部逐项 `assert_close` 通过 |
| 输出与梯度覆盖 | 完整小模块输出、输入梯度及模块全部参数梯度；未做采样或容差放宽 |
| 执行计数 | No-recompute：1/0/0；On-demand：1/0/1；Proactive：1/1/0（原始Forward/前置Replay/Backward Replay） |
| 全部测试 | 16 passed；最终日志见附录。中间测试夹具/可选进程组错误修复后复跑通过 |

Dropout测试验证了提前 replay 恢复原 Forward RNG、随后恢复调用者 RNG，并与 On-demand 输入梯度一致。
参数/输入原地更新通过 version counter拒绝。使用 `.data` 绕过 version counter 的写入不受支持。

## 4. 性能评估与定量测量

| 配置 | 峰值 allocated (B) | 峰值 reserved (B) | 原始Forward / 前置Replay / Backward Replay | 相对重复No-recompute均值 |
|---|---:|---:|---:|---:|
| No-recompute，重复1 | 17,061,376 | 23,068,672 | 1 / 0 / 0 | — |
| No-recompute，重复2 | 17,059,328 | 23,068,672 | 1 / 0 / 0 | — |
| No-recompute均值 ± 标准差 (n=2) | 17,060,352 ± 1,024 | 23,068,672 ± 0 | 1 / 0 / 0 | 基线 |
| On-demand (n=1) | 17,061,376 | 23,068,672 | 1 / 0 / 1 | allocated +0.006%；reserved 0% |
| Proactive (n=1) | 17,061,376 | 23,068,672 | 1 / 1 / 0 | allocated +0.006%；reserved 0% |

| 缓存观测 | 值 |
|---|---:|
| 输出storage | 1,920 B |
| 去重输出及 Autograd Saved Tensor storage | 5,632 B |
| Prefetch期间 allocated 当前值 | 17,052,160 → 17,054,208 B（增加2,048 B） |
| Prefetch期间 reserved 当前值 | 23,068,672 → 23,068,672 B |
| 缓存存活时间 | 0.00188 s（单次样本） |

- 该微型测试的 allocator峰值没有显示 Proactive 高于 On-demand；缓存事件仍直接记录到 5,632 B保留图状态。
- CUDA allocator峰值包含进程内其他活动，Saved Tensor storage是逻辑去重估算；二者不能简单相加或解释为训练级净额外显存。
- 本轮不报告吞吐或加速比。

## 5. 深度分析与机制探讨

**直接实测：**Proactive QKV forward hook计数为原始Forward一次、prefetch一次、Backward阶段没有第三次调用。事件顺序为 `forward_complete → prefetch_start → prefetch_complete → cache_hit → cache_consumed → released`。输出、输入梯度、全部参数梯度和 SGD 更新参数与两种基线通过相同默认容差。

**实现机制：**QKV投影沿用已有 `attn_qkv` 边界，以 `apply_module(self.linear_qkv)` 作为 replay函数。实例将QKV参数作为自定义 Autograd Function显式输入，并在内部对缓存图执行真实的上游梯度反传；输入梯度返回外层，参数梯度由图累积。传入参数使被冻结输入也能保持输出 Autograd连接。

Prefetch失败默认 terminal，避免未知副作用后静默重放；`allow_prefetch_fallback=True` 是调用者对 replay纯度的明确声明。检测覆盖输入、参数和 module buffers 的 version counter，不可能侦测任意 Python外部副作用。TE融合算子的反向状态、跨rank执行次序仍是待验证假设。

## 6. 局限性、决策建议与后续工作

- 已接入仅为 local `ColumnParallelLinear` QKV；没有接入 TE QKV、FC1或整模型训练。
- 当前支持 tensor/None输出；输入须为 Tensor。Dropout RNG、冻结输入、共享参数和输入别名已有单 GPU测试。
- 参数/输入 `.data` 写入可绕过 version counter，应视为不支持；调用闭包有外部副作用时不能打开安全回退标志。
- 生命周期测试覆盖5次顺序step及反序交错实例，但未模拟 pipeline排程、未完成实例跨 step清理政策或长时间显存曲线。
- 单卡 FP32小张量结果不能外推到 Qwen3-4B、BF16、TP通信、TE融合或真实 allocator压力。

后续建议：先审阅本地机制与报告；下一步专门验证 TE checkpoint/autograd状态边界，再增加多 micro-batch实例调度和显存预算。进入 TP2/DP4 前需验证所有 rank实例顺序一致、没有 rank间 collective顺序变化、参数分片版本各自受保护，并测量并发缓存峰值。按本轮授权，未运行多 GPU或通信重叠实验。

## 7. 附录：数据溯源与档案索引

| 证据 | 文件/字段 |
|---|---|
| 原型状态机、参数跟踪、Registry API | `Megatron-LM-layerwise/megatron/core/tensor_parallel/random.py`：`ProactiveCheckpoint`、`prefetch_recompute`、`pending_recompute_checkpoints`、`get_recompute_checkpoint` |
| QKV边界接入与支持限制 | `Megatron-LM-layerwise/megatron/core/transformer/attention.py`：`SelfAttention.get_query_key_value_tensors` |
| 显式默认关闭开关及验证 | `Megatron-LM-layerwise/megatron/core/transformer/transformer_config.py`：`proactive_recompute` |
| P0单 GPU验证 | `Megatron-LM-layerwise/tests/unit_tests/tensor_parallel/test_proactive_checkpoint.py` |
| P1/P2真实QKV验证 | `Megatron-LM-layerwise/tests/unit_tests/transformer/test_proactive_qkv.py` |
| 初始完整测试输出 | `pytest.log`；16 passed |
| 完整最终测试输出 | `final_pytest.log`；最终结果 16 passed |
| 中间失败记录 | `failure_notes.md`；`pytest_attempt_before_step_lifecycle_fix.log`、`pytest_attempt_incomplete_dp_group.log` |
| QKV计数与显存 | `qkv_metrics.json`：三路径replay计数、allocated/reserved当前值与峰值 |
| Prefetch事件 | `qkv_proactive_events.json`：checkpoint ID、层/模块、cache hit/consume/release、storage字节、存活时间 |
| 设计补充 | `Megatron-LM-layerwise/docs/research/proactive_recompute_design.md` |
