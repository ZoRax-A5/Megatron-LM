# P9：PP1 的双 micro-batch 调度窗口

S0 沿用逐个 Forward / Backward。启用 `--cross-microbatch-recompute-schedule` 后，PP1 使用 FIFO 的双 micro-batch 窗口，每次保留至多两个 loss 图。S1 单独启用调度开关。S2 另外启用 `--proactive-recompute`，由第二个 Forward 的 QKV 调用边界为前一个 micro-batch 的同层 QKV 提交 Replay。两模式共用同一调度分支。关闭调度开关时，当前 micro-batch Forward 结束后的主动重计算仍由原有选项控制。

DP 多卡时仅对原本位于 `no_sync` 内的 micro-batch 配对，末尾孤立成员按原顺序运行，最终 micro-batch 的 Forward 和 Backward 仍在 `no_sync` 外。DP1 有偶数个 micro-batch 时允许最后两个形成窗口，末次 Forward 与末次 Backward 在 `no_sync` 外，倒数第二次 Backward 仍受 `no_sync` 保护。这使 DP1 的两个 micro-batch 用例可以覆盖位置 B。Forward、Backward 各自保持原顺序，loss 缩放、梯度 finalize 和优化器更新由现有流程处理。

调度上下文传入 step、当前 micro-batch 与前驱 micro-batch。按 step、前驱 micro-batch、层和模块类型匹配已完成 Forward 的句柄，沿用输入/参数版本检查、RNG 状态恢复、TP 同序校验、容量限制与一次性消费路径。无预取结果的句柄走 Backward 内的按需重放。成功训练步结束前检查注册表。当前范围为 PP1/CP1/EP1 的 dense TE 训练，FP8/FP4、CUDA Graph 与 CPU offload 受配置校验限制。

完整训练实验使用原有逐 rank 验证器，在优化器前读取所有归约梯度分片，并在更新后读取全部模型参数。附加观察器只记录 QKV 调用、checkpoint 生命周期、训练步显存和梯度通信调用点。CUDA Event 在独立时间线采集中启用。
