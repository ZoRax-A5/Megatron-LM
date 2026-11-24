# P11 原生 PP1 的梯度桶触发方案

先用 P10 归档和原生 C0 的 NSYS 时间线构造任务窗口。候选层为 1、2、18，目标 micro-batch 是第 8 个。输入就绪的保守上界取原始 QKV 首个 GPU kernel，截止点以 C0 中原按需 Replay 首个 kernel 近似。只有在梯度桶的 host 提交返回后、对应真实 RS GPU kernel 尚未结束且目标未消费的区间才保留为候选。候选数据按 rank 记录。

选第 1 层及两个触发点。0 基 dispatch 0 是首桶，dispatch 35 是第 36 桶。C1 仍用 PP1 原生顺序。只有最终 micro-batch 第 1 层 QKV 建立 proactive 句柄，其它实例用原按需 checkpoint。持有图的上限为一个。

在最终 Backward 的作用域向 DDP bucket groups 临时安装一个 Python host 钩子。原来的 `register_grad_ready()` 在普通梯度钩子内认定桶就绪，提交异步 RS 后返回，才调用 P11 钩子。钩子并非 NCCL completion callback，且在通信 coalescing 区间外。它不调用 Autograd backward，只通过原有 Prefetch 的 GradMode 和 RNG 守卫创建 BF16 TE Replay 图。目标投影的 SP 与 TP overlap 关闭，Replay 本身无 TP collective。禁止在该路径引入原 P9 的 TP 校验 collective。反向消费保持在原 checkpoint 位置。

首步的桶就绪计数尚未建立，`finish_grad_sync()` 中的提交不会进入 P11 钩子。目标不存在、已消费或仍未遇到指定桶时保持按需路径。版本或 Replay 错误由已有机制终止。原生异步 Distributed Optimizer RS 是启用该实验的运行约束。

P10 的 Saved Tensor 包装问题在目标消费经过最终梯度同步时会影响 TE 通知 DDP。P11 复用之前未提交的按目标禁用统计钩子补丁。缓存成本以 CUDA allocated 局部增量与整步峰值分别测量。P11 所有 GPU 执行复用主计算 Stream 与 Megatron 原生 NCCL Stream。
