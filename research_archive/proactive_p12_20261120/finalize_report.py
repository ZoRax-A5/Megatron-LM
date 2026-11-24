"""Render the final P12 tables after all matched runs have passed."""
import json
from pathlib import Path
import statistics as st

p = Path(__file__).resolve().parent
data=json.loads((p/"performance_summary.json").read_text())
comp=json.loads((p/"comparison.json").read_text())
assert len(comp)==4 and all(r["checks"]==48 and not r["mismatches"] for r in comp.values())
assert len(data["runs"])==4 and all(len(v)==5 for v in data["runs"].values())
want=("C0","Cnoop","C1","Clate")
ex = {m:json.loads((p/f"p12_4b_{m}_trace_conn1_exposure.json").read_text()) for m in want}
life=json.loads((p/"timeline_audit.json").read_text())

def vals(m,k):
    return [r[k] for r in data["runs"][m].values()]

def interval(x,digits=3):
    return f"{st.median(x):.{digits}f}［{min(x):.{digits}f}, {max(x):.{digits}f}］"

perf=["| 模式 | 整步中位数 ms［五轮范围］ | 峰值 allocated GiB［范围］ | 峰值 reserved GiB［范围］ |",
      "|---|---:|---:|---:|"]
for mode in want:
    perf.append(f"| {mode} | {interval(vals(mode,'median_ms'),1)} | "
      f"{interval([n/2**30 for n in vals(mode,'allocated')])} | "
      f"{interval([n/2**30 for n in vals(mode,'reserved')])} |")
pairs=["| 配对比较 | 五轮整步差值 ms | 中位差 ms | 中位相对差 | 峰值 allocated 中位差 |",
       "|---|---|---:|---:|---:|"]
for name, rows in data["paired"].items():
    a,b=name.split("-")
    diff=[r["delta_ms"] for r in rows]
    pct=[r["delta_ms"]/data["runs"][b][str(r["run"])]["median_ms"]*100 for r in rows]
    alloc=[r["allocated_delta"]/2**20 for r in rows]
    pairs.append(f"| {name} | {' / '.join(f'{v:+.1f}' for v in diff)} | {st.median(diff):+.1f} | "
                 f"{st.median(pct):+.2f}% | {st.median(alloc):+.2f} MiB |")
c1_perf=[r["delta_ms"] for r in data["paired"]["C1-C0"]]
c1_net=[r["delta_ms"] for r in data["paired"]["C1-Cnoop"]]
noop=[r["delta_ms"] for r in data["paired"]["Cnoop-C0"]]
late=[r["delta_ms"] for r in data["paired"]["Clate-C1"]]
summary=f"""**P12 对照与机制验证 PASS。以现有 DP 固定桶方案继续扩大 Replay 数量，性能判断为 NO-GO。** C1 将 QKV 前向 Replay 与已被原生 Backward 覆盖的早一桶通信并发。仍然暴露的终端 DP RS 约 3.05–3.08 ms，来自必须等第 1 层梯度的末桶。第 36 桶从未进入这段尾部。

- 原位置单个 QKV Replay 只占约 0.168 ms。并发时同一 QKV 的 Kernel 并集升至约 0.326 ms，观察到的交集不能直接计为收回的时间。
- 五轮独立运行中，C1 相对 C0 的整步差值范围为 {min(c1_perf):+.1f} 至 {max(c1_perf):+.1f} ms，相对空钩子为 {min(c1_net):+.1f} 至 {max(c1_net):+.1f} ms。C1 五轮都快于 C0，但相对 Cnoop 只有两轮更快，中位差为 {st.median(c1_net):+.1f} ms。该幅度无法由单个 Replay 的计算消除解释。
- 第 35 桶把 Replay 完成至消费的 GPU 距离拉长至 5.97 ms，第 36 桶为 1.39 ms。两处缓存成本都是 22.01 MiB/rank，每 rank 最多同时 1 图。
- 8 rank 完整数值通过。离线考察第 1、2、18 层之后，未开展多任务 GPU 运行或 TP/SP 改造。"""
timing = f"""无 Profiler，每轮取第 6–10 步的整步中位数。峰值指标先对每轮的全部 rank 取最大值，再汇总五轮。两组运行的配对依据预先分配的轮次。

{chr(10).join(perf)}

{chr(10).join(pairs)}

空钩子相对 C0 的轮次差值为 {min(noop):+.1f} 至 {max(noop):+.1f} ms，中位差 {st.median(noop):+.1f} ms。诊断中仅包围空返回的 NVTX 范围中位数为 1.82 µs，该值不含注册、查找、目标图回退及测量之外的其它代码。整步对照的波动比微秒级开销大得多，无法从整步数据精确分离每个钩子的成本。C1 相对 C0 的配对中位差为 {st.median(c1_perf):+.1f} ms，相对 Cnoop 为 {st.median(c1_net):+.1f} ms。Clate 相对 C1 中位差 {st.median(late):+.1f} ms，两个位置的缓存局部代价均为 22.01 MiB/rank。数据保留了 C1 相对原路径的五轮同向变化，但空钩子对照未复现稳定的净收益。"""

trace=["| 模式 | 目标 Replay Kernel 中位数 ms | 目标与选定 RS 的实际交集 ms［各 rank 范围］ | Replay 完成至消费中位数 ms | 跨 rank 终端 RS 尾部 ms | 诊断 GPU step 包络中位数 ms |",
       "|---|---:|---:|---:|---:|---:|"]
for m in want:
    x=ex[m]["summary"]
    ov=x["qkv_dp_actual_ms"]
    residency=f"{life[m]['summary']['lifetime_gpu_ms']['median']:.3f}" if m in ("C1","Clate") else "不适用"
    trace.append(f"| {m} | {x['qkv_math_ms']['median']:.3f} | {ov['median']:.3f}［{ov['min']:.3f}, {ov['max']:.3f}］ | "
                 f"{residency} | {ex[m]['cross_rank_wall_tail_ms']:.3f} | "
                 f"{x['step_profiler_gpu_envelope_ms']['median']:.1f} |")
timeline=f"""### P12 对照的关键路径证据

{chr(10).join(trace)}

四组首个优化器 GPU Kernel 均晚于各自最后一个 DP RS 完成。C0 两者差值的跨 rank 中位数为 11.59 ms，此区间还经过梯度 finalize、后续同步和优化器准备。第 35/36 桶均在最终 Backward 结束前完成，因此其并发不能直接消除后面的末桶服务。若提前执行能压缩最后一桶提交前的关键计算，末桶仍可能更早启动。对本目标而言，单靠消除原 Replay 计算的理想改善量级仍约 0.168 ms。四组在梯度 finalize 中各有 37 次 CUDA 等待 API 提交/rank。诊断单步的 GPU 包络有显著波动，例如空钩子也出现了约 70 ms 的缩短。整步收益据无 Profiler 的配对结果判断。

新 trace 中 C1 和 Clate 各有 8/8 rank 看到交集，各有 6/8 rank 的两个目标 Kernel 完全落入同一个 RS。C1 在首个 Replay Kernel 开始时，目标 RS 的剩余时间为 0.208–0.959 ms，中位数 0.777 ms。C0 的目标 QKV API 入口至首次 GPU 执行为 0.343 ms 中位数，该数包含 CPU 构建和 CUDA 排队。两个 QKV Kernel 之间的 GPU 间隔仅约 0.001 ms。C1 在原位置消除的两个 Kernel 合计约 0.168 ms，但提前位置的计算仍在原生 Autograd 计算 Stream 上运行。

在 C0 的相应原生 Replay 计算 Stream 上，第 35/36 桶候选区间内最长的无 Kernel 区段中位数分别为 1.20 和 1.31 ms，几何长度足以容纳实测 Replay。这类区段还可能存在未记录的 CUDA 等待，不能用来证明所有间隙可调度。C1 与 Clate 的实测已验证这两个具体触发点的合法执行。

分开看，C0 的 RMSNorm 和 GEMM 的 Kernel 中位数分别为 33.76 和 134.54 µs，C1 为 58.93 和 269.97 µs。两类 Kernel 延长与 NCCL 同时占用 GPU 资源相符。该对照无法进一步断言是哪一种 SM 或内存资源造成的。"""
multi="""| 末 micro-batch 目标层 | 最新观测可用梯度桶 | 该桶返回至原 Replay 中位数 | P12 C0 原 Replay 中位数 |
|---|---:|---:|---:|
| 第 1 层 | 36 | 1.671 ms | 约 0.168 ms |
| 第 2 层 | 35 | 1.658 ms | 约 0.168 ms |
| 第 18 层 | 19 | 1.668 ms | 约 0.168 ms |

候选筛选同时要求原输入已就绪、目标尚未消费，以及同 rank 真实 RS 窗口的剩余长度覆盖原始 Replay。第 18 层在 P11 采样中已有自然重叠，在 P12 C0 采样中为零，说明候选机会随 rank 到达变化。当前三个 Replay 的原计算全消除上界为每 rank 0.503–0.505 ms，P11 的基线估计为 0.503–0.676 ms。两图若同时驻留，按单图实测成本估计约 44.02 MiB/rank。

**多任务 GPU 验证 NO-GO。** 上述可移动点有助于扩大可观察的 Kernel 交集，但选出的桶均不能覆盖依赖目标梯度的终端 RS。低于 0.68 ms 的乐观计算上界小于本轮整步波动，缺少其能缩短关键路径的证据。依据阶段门槛，本轮停在离线候选验证。"""
tp="""P12 C0 每 rank 观察到 608 个 Forward All-Reduce、584 个 Backward All-Reduce，并有 32 个 Forward 数据 Broadcast。单 All-Reduce 跨 rank/调用的中位数分别为 124.46 和 127.86 µs。模型通信总驻留的跨 rank 中位数为 310.83 ms，其中 80.32 ms 与普通计算 Kernel 有实际交集。剩余驻留包含运算等待和 rank 到达偏斜，没有因果依据把它全部算作暴露时间。

用实际 TP 组成员和相同 collective 顺序匹配两端，在末 micro-batch Backward 内取两个 rank 都已返回提交、都仍在执行通信、且目标仍有合法输入和消费期限的共同区间。枚举得到 717 个正区间，计数包含不同层和 TP 组组合。12 个区间可容纳所属 TP 组两端的基线 Replay，最长 0.198 ms。没有同一候选 ordinal 能让四个 TP 组都完整覆盖。模型 TP 同步还会把通信等待排入依赖计算 Stream，单凭这些区间不能认为同一 Stream 已能安全提前插入。

**TP 暂不比 DP 更值得直接实现原型。** 若继续调查，先在一个有共同窗口的高层 All-Reduce 上加最小 CUDA 等待依赖观测，确认低层已保存 QKV 输入独立于该归约输出，再评估能否把纯计算提交放在合法的 Stream 等待之前。保持原 collective 调用次序，具备 P11 同等级的全量梯度对照再扩展。SP 与细粒度 QKV checkpoint 现有配置互斥。适配需处理 Replay 自身的 AG/RS、保存输入的分片语义和跨 rank 次序，本轮仅保留此分析。"""
f=p/"EXPERIMENT_REPORT.md"
s=f.read_text().replace("<!-- SUMMARY -->",summary).replace("<!-- PERFORMANCE -->",timing)
s=s.replace("<!-- CORRECTNESS -->",'''NSYS 审计验证各组逐 rank 的 TP 模型通信 Kernel 提交签名一致，均有 37 次原生异步 DP RS，桶顺序与 C0 一致。两个主动模式的目标执行次数为 1 次原 Forward、1 次主动 Replay、0 次 Backward Replay。Cnoop 命中钩子但没有提前产生 Kernel，目标仍在原位置重放。三组实验路径首步按需回退，后续所有目标均按规划触发并释放。''')
s=s.replace("<!-- TIMELINE -->",timeline).replace("<!-- MULTI -->",multi).replace("<!-- TP -->",tp)
s=s.replace("<!-- LIMITS -->",'''保留 P11 晚触发机制供后续研究，但不以增加 Replay 数量作为当前加速方向。后续若要投入，优先找能减少优化器实际等待的结构性变化，再在固定 batch 和通信参数下验证；同时比较 C0、Cnoop 和目标方案的 GPU 依赖与无 Profiler 整步数据。''')
s=s.replace("| 重复性能 |","| 关键路径图 | `critical_path_timeline.png`，由 `plot.py` 从真实 GPU Kernel 时间戳生成。Replay 生命周期及通信提交审计为 `timeline_audit.json`，固定 trace 的计算 Stream 几何间隙见 `fixed_trace_slack.json`，finalize 等待 API 见 `cuda_grad_finish_wait_api.json`。 |\n| 重复性能 |")
assert "<!--" not in s
f.write_text(s)
