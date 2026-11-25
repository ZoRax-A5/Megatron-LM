# 激活重计算研究档案

本目录汇集 Megatron 细粒度重计算与主动重计算实验，供在线审阅和规划后续实验。

## 阅读顺序

| 阶段 | 报告 | 主要内容 |
|---|---|---|
| Qwen3-4B 基线 | [AB 报告](qwen4b_ab_20261118/EXPERIMENT_REPORT.md) | 真实训练基线与并行通信 |
| 细粒度重计算 | [阶段 C 报告](qwen4b_c_20261118/EXPERIMENT_REPORT.md) | QKV、FC1 与 DP 通信 |
| 主动重计算原型 | [原型](proactive_recompute_20261119/EXPERIMENT_REPORT.md)、[QKV](proactive_recompute_qkv_20261119/EXPERIMENT_REPORT.md) | 计算图复用与单卡验证 |
| 真实 TE 训练 | [训练验证](proactive_qwen_training_20261119/EXPERIMENT_REPORT.md) | BF16、TE、TP2/DP4 与训练稳定性 |
| 跨 micro-batch | [P8](proactive_cross_microbatch_20261119/EXPERIMENT_REPORT.md)、[P9](proactive_p9_20261119/EXPERIMENT_REPORT.md) | 调度原型及正式训练接入 |
| 通信并发 | [P10](proactive_p10_20261120/EXPERIMENT_REPORT.md)、[P11](proactive_p11_20261120/EXPERIMENT_REPORT.md) | GPU Kernel 并发与原生 PP1 |
| 收益分析 | [P12](proactive_p12_20261120/EXPERIMENT_REPORT.md) | 空钩子对照、关键路径与候选窗口 |
| TP 依赖分析 | [P13](proactive_p13_20261124/EXPERIMENT_REPORT.md)、[P14](proactive_p14_20261124/EXPERIMENT_REPORT.md)、[P15](proactive_p15_20261124/EXPERIMENT_REPORT.md) | Event 等待、张量依赖与 TP 路线决策 |

## 档案范围

保存实验报告、设计文档、主要图表、摘要 JSON、逐 rank 验证记录、配置和实验脚本。报告附录记录的是实验现场的原始路径，部分文件没有复制到本仓库。原始文件保存在工作区的 `qwen3_bench/results/<实验目录>/`。

大型 NSYS 原始 trace、模型权重、完整梯度和参数快照未上传。完整张量快照包含每个被验证张量的所有元素，可以逐元素重新比较，单次实验可达数百 GB。仓库中保留的验证结果、哈希及汇总指标便于审阅，但不能替代对原始张量或 trace 的重新分析。

实验脚本保留原始版本，可能依赖工作区布局、权重与环境，不保证从此目录直接运行。实验结果的适用范围、统计口径和失败记录以各报告为准。

## 后续同步约定

每个阶段在独立结果目录完成实验后，更新此索引，并将报告、关键数据、图片、必要脚本和源码修改纳入同一分支提交。报告说明提交基线、完整实验配置、验证覆盖范围、关键数据的原始文件及未上传的大文件位置。提交并推送后可从 GitHub 分支阅读。自动同步尚未配置，后续每阶段需执行提交和推送。

## P17：MoE EP 基线与后续输入

[P17 实验报告](proactive_p17_moe_ep_20261124/EXPERIMENT_REPORT.md) 建立了单机八卡 EP2/4/8 mock-data 基线，识别了 Dispatch/Combine 的实际通信并测量 `moe_fc1` 成本。阶段判定 **PARTIAL（未达到 PASS）**：独立运行的完整参数严格对照未通过，真实关键路径等待尚未知。P18 暂缓；候选输入是 EP8 token Dispatch/Combine 与 expert FC1 Replay，须先完成数值对照与可证明的依赖、等待验证。

[P17 补充验收（2026-11-25）](proactive_p17_moe_ep_20261124/followup_20261125/EXPERIMENT_REPORT.md)：确定性 EP2 首步自对照与 `moe_fc1` 完整张量对照逐字节一致；稳态 EP8 样本识别到通信流与计算流事件依赖。0.43B 放大模型未见明确 `moe_fc1` 峰值显存收益；其余重计算额外步时未能因果分解，合法 Replay 窗口尚未证明，P17 仍为 PARTIAL、P18 暂为 NO-GO。
