"""Insert validated repeat-run data into the P11 report."""
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
d = json.loads((root/"summary.json").read_text())
assert len(d["runs"])==4 and all(len(v)==3 for v in d["runs"].values())
assert all(not x["mismatches"] for x in json.loads((root/"comparison.json").read_text()).values())
labels = {"S0":"C0", "C1early":"C1-首桶", "C1late":"C1-第36桶", "same":"C2"}
med = statistics.median

def row_interval(values, digits):
    return (f"{med(values):.{digits}f}［{min(values):.{digits}f}, "
            f"{max(values):.{digits}f}］")

def max_gib(mode, key):
    return [r[key]/2**30 for r in d["runs"][mode].values()]

a = ["| 模式 | 整步中位数 ms［三轮范围］ | 最大 allocated GiB［三轮范围］ | 最大 reserved GiB［三轮范围］ |",
     "|---|---:|---:|---:|"]
for mode,label in labels.items():
    a.append("| " + " | ".join((label, row_interval([r["steady_ms"]["median"] for r in d["runs"][mode].values()], 1),
             row_interval(max_gib(mode,"max_allocated"),3),
             row_interval(max_gib(mode,"max_reserved"),3)))+" |")
b = ["| 与同轮 C0 对比 | 三轮整步差值 ms | 中位差 ms | 中位相对差 | 最大 allocated 差值中位数 |",
     "|---|---|---:|---:|---:|"]
for mode in ("C1early","C1late","same"):
    runs = d["paired"][mode]
    dt = [p["delta_ms"] for p in runs.values()]
    pct = [p["delta_ms"]/d["runs"]["S0"][r]["steady_ms"]["median"]*100 for r,p in runs.items()]
    dm = [p["allocated_delta"]/2**20 for p in runs.values()]
    b.append(f"| {labels[mode]} − C0 | {' / '.join(f'{v:+.1f}' for v in dt)} | {med(dt):+.1f} | {med(pct):+.2f}% | {med(dm):+.2f} MiB |")
late = [r["delta_ms"] for r in d["paired"]["C1late"].values()]
early = [r["delta_ms"] for r in d["paired"]["C1early"].values()]
late_vs_c2 = [d["runs"]["same"][r]["max_allocated"]-d["runs"]["C1late"][r]["max_allocated"]
              for r in d["runs"]["C1late"]]
late_vs_c2 = [v/2**30 for v in late_vs_c2]

text = f"""每组独立启动 3 次，单次 12 步。每轮先取第 6–12 步整步时长的中位数，再比较三个独立运行。显存为每轮所有 rank 的整次峰值最大值。

{chr(10).join(a)}

{chr(10).join(b)}

C1-第36桶的整步变化为 {min(late):+.1f} 至 {max(late):+.1f} ms，中位差 {med(late):+.1f} ms。C1-首桶的中位差为 {med(early):+.1f} ms。晚触发相对 C2 降低峰值 allocated 中位数 {med(late_vs_c2):.3f} GiB，主要来自保留原生 PP1 调度。C1 与 C0 的峰值接近。每个 C1 运行在稳态的每 rank 每步只持有一个主动图，首步按需回退，无跳步或 NaN。"""
summary = f"""**P11 功能 PASS。整步加速未得到可重复证据。** 原生 PP1 可以在最后一个 micro-batch 的 Backward 内，利用已提交的异步 DP RS 提前执行第 1 层 QKV Replay。两种触发点均通过正确性验证。第 36 桶是本轮的优先实验方案，因为它在采集步缩短了缓存驻留并覆盖了全部 rank 的 RS 窗口。

- 第 36 桶方案在 8/8 rank 上观测到真实 GPU Kernel 交集，范围 0.238–0.337 ms。
- 全部 8 rank、连续 3 步的完整梯度分片、更新参数及 loss 与 C0 一致。每个成功实例执行 **1 次原始 Forward、1 次主动 Replay、0 次 Backward Replay**。
- 晚触发的 GPU Replay 完成至消费中位数为 1.387 ms，对照 C2 为 152.512 ms。整步 allocated 峰值比 C2 低约 {med(late_vs_c2):.2f} GiB。
- 晚触发对 C0 的三轮整步时间差为 {'、'.join(f'{v:+.1f}' for v in late)} ms。现有数据支持保留原型用于调度研究。"""

p = root/"EXPERIMENT_REPORT.md"
s = p.read_text().replace("<!-- SUMMARY -->",summary).replace("<!-- PERFORMANCE -->",text)
s = s.replace("<!-- LIMITS -->",'''继续研究时优先使用原生 PP1 的晚桶触发，进一步检查桶分组和目标消费之间的稳定性，区分等待被隐藏的时间与 Replay 在 NCCL 同时执行时增加的时延。本轮没有足够性能依据开启默认使用。''')
if (root/"unit.log").exists():
    report = (root/"unit.log").read_text()
    assert "19 passed" in report
    s += "| P9/Checkpoint 回归 | `unit.log`，现有两份单 GPU 用例共 19 项通过。 |\n"
assert "<!--" not in s
p.write_text(s)
