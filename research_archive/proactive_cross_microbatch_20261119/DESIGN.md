# P7：跨 micro-batch 主动重计算调度分析

## 实际顺序与窗口

在常规 PP1 分支，前 N−1 个 micro-batch 在 `no_sync` 内各自按 `forward_step(i) → backward_step(i)` 执行，最后一个在 `no_sync` 外按相同顺序执行以同步梯度；全部结束后执行梯度 finalize。此前配置 GBS32、MBS1、DP4，每 rank 每 step 为 32/(1×4)=8 个 micro-batch。因此 B 的 forward 开始时，A 的 backward 已经完成；未改变这一顺序不能构造要求的窗口。位置 A 的现有 hook 在 `forward_step` 计算 loss 后预取当前 micro-batch，不是跨 micro-batch 触发。

`pending_recompute_checkpoints()` 只返回状态为 `forward_complete` 的注册实例；句柄 metadata 含 step、microbatch、layer、module 和 rank。Forward 已完成而 Backward 未消费期间，可基于真实身份查找该实例；输入与参数版本和 RNG 快照仍须通过现有检查。TP2 下各 rank 必须同序执行已有 prefetch 身份握手；DP 梯度同步必须保持最后 micro-batch 的同步边界，异常应终止训练，不能回滚不完整的 main_grad 后继续。

## 最小受控实验方案

单独增加默认关闭的两 micro-batch 实验调度：`F(A) → F(B，指定真实模块 hook 内触发 replay(A)) → Bwd(A) → Bwd(B)`；其余 micro-batch 保持原有顺序。把 A、B 都放入已有 no_sync 区间且不使用最后 micro-batch，最后一个仍在 no_sync 外完成以维持归约边界。触发点需要位于 B 的实际模块 forward 内并由其真实 step/microbatch context 标识；对同 TP 组采用固定 layer 和一致身份，禁止局部就绪决定调用。必须保留两个 loss 图直至各自 backward，记录其额外 activation 峰值；参数不得在两者之间更新。此调度改变 forward/backward 交错顺序，可能影响 TE 首 micro-batch 标志、RNG、梯度累积次序及 DDP hook；应逐项验证，不能预设数值等价。

在 B 的至少两个真实模块边界插入同 stream 可选 hook，记录 CPU 单调时钟、CUDA event 及 NVTX：A forward 完成、B 阶段起止、A replay 起止、A backward 消费和释放。event 同步后核对 GPU 事件顺序，区分 host 提交与 GPU 执行；顺序交错不表示 kernel 并发。以 On-demand 和位置 A 同调度、同训练数据与权重为对照，先单 GPU，再 TP2，再 TP2/DP4 全 rank 全量有效梯度和参数；覆盖缓存上限、错误身份、重复调用与 backward 异常终止。通过后再进行 Qwen 有限训练和显存对照。

## 当前状态

这是静态源码分析及实验设计，未实现该调度或执行 P8–P11。特别是未验证 TE 状态、GPU 时间线、跨 micro-batch 数值等价和显存成本，不能宣称位置 B PASS。原有工作树改动已在本目录保存 tracked patch 和未跟踪测试副本，未覆盖历史结果。

## P8 原型结果

独立测试以 B 的 TE QKV forward pre-hook 触发 A 的 Replay，CUDA Event 顺序与单层完整梯度比较通过。该实现存在于显式运行的测试中，尚未接入训练 schedule；详细范围、内存和时间线见本目录 `EXPERIMENT_REPORT.md`。
