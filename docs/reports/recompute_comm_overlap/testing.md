# 重计算与通信重叠：测试报告

## 测试环境

- 日期：2026-09-14
- GPU：2 × NVIDIA A100 80GB PCIe
- PyTorch：2.5.1+cu124
- NCCL：2.21.5
- 可用 GPU 数：2；8 卡和跨节点验证未执行
- 外部环境未安装 Transformer Engine、Apex、einops
- 仓库 `uv.lock` 在 line 2183 缺少 package version，因此无法使用 `uv run`；测试使用已有隔离 Python 环境

## 单元与调度测试

| 范围 | 结果 | 覆盖 |
| --- | ---: | --- |
| EP 调度与配置 | 5 passed | 默认顺序、通信优先顺序、选择性重计算约束、MTP 回归 |
| PP 配置/调度选择 | 3 passed | 默认关闭、双开关、非法单开关 |
| DDP/FSDP 配置和预取 | 3 passed | 默认关闭、backward-order、单桶限制、梯度开关映射 |
| TP 协调器 | 9 passed | launch→recompute→wait、fallback、异常 drain、单次消费、非法配置 |

## 双卡 NCCL 数值测试

TP 测试在两个 rank 上均通过：

- sequence-parallel All-Gather 前向数值正确；
- All-Gather 反向 Reduce-Scatter 梯度正确；
- Reduce-Scatter 前向数值正确；
- Reduce-Scatter 反向 All-Gather 梯度正确；
- 原生 `linear_with_frozen_weight` 接入和回调消费正确。

DDP 现有 `test_grad_sync` 的单优化器实例矩阵在两个 rank 上各 8 passed，覆盖：

- All-Reduce 与 distributed-optimizer Reduce-Scatter；
- overlap on/off；
- collective 内平均 on/off；
- 梯度数值与桶就绪协议。

多优化器实例的额外 8 组在测试初始化阶段因环境缺少 `einops` 失败，尚未进入本次改动路径。这是环境依赖限制，不计为实现通过。

## 双卡硬件共存 profiling

命令：

```bash
torchrun --standalone --nproc-per-node=2 \
  tools/recompute_comm_overlap_benchmark.py \
  --message-mb 64 --matrix-size 4096 --compute-iters 4 \
  --warmup 2 --iters 5
```

每项取双 rank 最大 wall time。隐藏比例为 `(sequential - concurrent) / min(comm, recompute)`；并发干扰为 `concurrent / max(comm, recompute) - 1`。

| 原语 | 通信 ms | 重计算 ms | 串行 ms | 并发 ms | 隐藏比例 | 并发干扰 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2P | 0.562 | 2.189 | 2.719 | 2.538 | 32.3% | 16.0% |
| All-Reduce | 0.508 | 2.346 | 2.794 | 2.691 | 20.4% | 14.7% |
| All-Gather | 0.610 | 2.355 | 2.875 | 2.644 | 38.0% | 12.3% |
| Reduce-Scatter | 0.551 | 2.390 | 2.865 | 2.704 | 29.1% | 13.2% |
| All-to-All | 0.547 | 2.371 | 2.865 | 2.650 | 39.4% | 11.8% |

在本机 A100 PCIe、该消息和 GEMM 配置下，干扰从低到高为 All-to-All、All-Gather、Reduce-Scatter、All-Reduce、P2P。排序只适用于此拓扑与工作负载；8×A100 NVLink/NVSwitch 上必须重测。

## 静态检查

- `isort`：对新增文件和本次实际修改 import 的文件执行并通过；未机械重排历史文件。
- `ruff check`：通过。
- `git diff --check`：通过。
- 聚焦 Python compile：通过。

## 尚需 8 卡验证的项目

2 卡足以验证 API、依赖顺序、buffer 生命周期、NCCL 数值与单机硬件争用，但不能替代以下验收：

1. EP size 4/8 下动态 token 分发、All-to-All 尾延迟和显存峰值。
2. PP 4/8 stage 的完整 warmup、steady、cooldown 与 virtual pipeline bubble。
3. TP 4/8 下环算法、Userbuffers/Transformer Engine 集成。
4. DP/FSDP 4/8 rank 的多桶并发、HSDP outer group 和多优化器实例。
5. 端到端 MoE loss parity、吞吐、step-time p50/p95 和 OOM 边界。

因此当前结论是“两卡功能与原语级实现完成并验证”，不是“八卡生产性能验收完成”。
