# Communication-aware activation recomputation

MoE training often needs activation recomputation to fit the model, but recomputation increases
the exposed backward compute time. Communication-aware recomputation places this extra compute in
independent EP, PP, DP/FSDP, or TP communication windows selected by measured hardware
coexistence. Every path is opt-in and disabled by default.

## EP All-to-All

The combined-1F1B MoE path splits each layer into attention, dispatch All-to-All, routed-expert
compute, and combine All-to-All nodes. With --recompute-comm-overlap, dispatch All-to-All is
launched before the previous microbatch's expert backward. Selective recomputation triggered in
that backward can then overlap the in-flight dispatch. The combine path is already
communication-first, so attention backward recomputation overlaps combine All-to-All.

The option is opt-in and only changes launch order. It does not silently add recomputation or
change the model graph.

    --overlap-moe-expert-parallel-comm
    --recompute-granularity selective
    --recompute-modules layernorm moe_act core_attn
    --moe-grouped-gemm
    --recompute-comm-overlap

moe_act requires grouped GEMM. Existing precision and CUDA graph restrictions on selective
recomputation still apply. Full-layer and whole-MoE recomputation are intentionally not accepted
by this schedule because they replay token routing and collectives and can deadlock or route a
different dynamic token layout.

## Implemented communication domains

| Domain | Primitive | Integration |
| --- | --- | --- |
| EP | dispatch/combine All-to-All | automatic in combined-1F1B |
| PP | P2P send/receive | automatic in interleaved pipeline |
| DP | gradient All-Reduce | automatic through asynchronous DDP buckets |
| DP | gradient Reduce-Scatter | automatic with the distributed optimizer |
| FSDP | parameter All-Gather | automatic backward-order one-group prefetch |
| TP | sequence All-Gather/Reduce-Scatter | native Linear integration with an explicit callback context |

PP additionally requires `--pp-recompute-comm-overlap`. Gradient synchronization requires
`--recompute-grad-sync-overlap`. Megatron-FSDP parameter prefetch requires
`--megatron-fsdp-recompute-ag-overlap`. The TP API and restrictions are documented in the usage
report linked below.

The TP coordinator intentionally does not infer an independent activation from the training graph.
The callback must not depend on the pending collective and must not issue a TP collective.
Transformer Engine Userbuffers are not supported by this coordinator.

## Choosing a communication target

Profile every candidate primitive with the same message size and representative recomputation
kernel. Rank candidates by concurrent slowdown relative to the slower standalone component:

    interference = concurrent_time / max(communication_only_time, recompute_only_time) - 1

Prefer a low-interference window that is also long enough to hide the selected recomputation.
Use `tools/recompute_comm_overlap_benchmark.py` to measure P2P, All-Reduce, All-Gather,
Reduce-Scatter, and All-to-All on the target topology. Results from two PCIe A100s must not be
assumed to hold on an eight-GPU NVLink/NVSwitch node.

## Dynamic expert-memory imbalance

Token imbalance creates both compute skew and unequal temporary activation memory. Megatron Core
already provides two complementary controls, so the recommended first implementation does not
need a new memory-dependent router objective:

* `--moe-router-enable-expert-bias` applies a slowly updated per-expert bias. It reduces persistent
  token skew without adding an auxiliary loss to the model objective.
* `--moe-expert-rank-capacity-factor` puts a hard upper bound on the temporary token buffer owned by
  an expert-parallel rank. It currently requires the HybridEP flex dispatcher and the Transformer
  Engine op fuser. The paged-stash runner detects an over-budget step and reruns it once without
  the capacity optimization, so tokens are not silently lost.

A memory-bounded configuration can therefore add:

    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend hybridep
    --use-transformer-engine-op-fuser
    --moe-expert-rank-capacity-factor <profiled-factor>

Choose the rank capacity factor from routed-token and peak-memory profiles, with enough reserve for
the attention activations introduced by the overlap schedule. This separates the two time scales:
dynamic expert bias handles long-term load balance, while rank capacity bounds per-step memory.

A future adaptive controller can track an exponential moving average of bytes per routed token and
free memory per expert rank, then update a rank-level congestion component of the expert bias with
hysteresis. That extension needs convergence evaluation because memory-dependent routing changes
the selected experts; the rank-capacity fallback above is the safer production baseline.

## Detailed reports

The implementation status, exact flags/API, test commands, two-GPU measurements, and deferred
eight-GPU acceptance work are recorded in:

* [Implementation report](../../reports/recompute_comm_overlap/implementation.md)
* [Usage report](../../reports/recompute_comm_overlap/usage.md)
* [Test report](../../reports/recompute_comm_overlap/testing.md)
