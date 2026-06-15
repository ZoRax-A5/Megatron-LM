# CUDA Compute and NCCL Communication Overlap

This directory is a small, standalone experiment for learning how GPU computation
and distributed communication can execute concurrently. It does not depend on
Megatron model code.

The example targets a single node with two or more NVIDIA GPUs, including A100.
It uses:

- PyTorch GEMMs as the computation workload.
- NCCL `all_reduce` or `all_gather` as the communication workload.
- ProcessGroupNCCL's internal communication stream.
- One optional user communication stream for explicit dependency management.
- CUDA events for cross-stream dependency management.
- NVTX ranges for inspection with Nsight Systems.

## Core Idea

PyTorch's ProcessGroupNCCL launches collectives on internal NCCL streams.
Calling a collective with `async_op=True` returns a `Work` handle before the
collective completes. Independent computation can then be enqueued before the
program waits for the communication result:

```python
work = torch.distributed.all_reduce(tensor, async_op=True)

# Enqueue work that does not consume tensor.
run_independent_gpu_computation()

# Wait at the latest point before tensor is consumed.
work.wait()
```

The timeline can then look like:

```text
compute stream:  |---------- GEMM computation ----------|
NCCL stream:     |------ NCCL communication ------|
```

Calling `work.wait()` immediately after the collective removes this scheduling
opportunity. `async_op=True` is necessary for this pattern, but it is not enough
if the program immediately introduces a dependency.

For custom CUDA operations, or when explicit stream dependencies are useful,
the same idea can be expressed with a user stream and event:

```python
comm_done = torch.cuda.Event()
comm_stream = torch.cuda.Stream()
default_stream = torch.cuda.current_stream()

with torch.cuda.stream(comm_stream):
    work = torch.distributed.all_reduce(tensor, async_op=True)
    work.wait()
    comm_done.record()

run_independent_gpu_computation()
default_stream.wait_event(comm_done)
```

`ProcessGroupNCCL` still owns the stream on which its NCCL kernel runs.
`work.wait()` links NCCL completion to the currently active user stream, and
the event links that user stream to the default compute stream.

Overlap is possible only if:

1. The collective is launched asynchronously.
2. There is no premature `work.wait()` or `torch.cuda.synchronize()`.
3. Communication and computation do not have a true data dependency.
4. The GPU and interconnect have enough resources to run both efficiently.

NCCL kernels consume GPU resources too, so overlap is not always free. A large
GEMM can reduce NCCL progress, and NCCL can reduce GEMM throughput.

## Run on A100

Use at least two GPUs:

```bash
NUM_GPUS=2 examples/communication_overlap/run.sh
```

The launcher uses `python3` by default. Override it when the container's
PyTorch environment uses another interpreter:

```bash
PYTHON=/opt/venv/bin/python NUM_GPUS=2 \
    examples/communication_overlap/run.sh
```

Run an all-gather experiment:

```bash
NUM_GPUS=2 examples/communication_overlap/run.sh \
    --collective all_gather \
    --comm-size-mb 128 \
    --matrix-size 4096 \
    --compute-repeats 8
```

The program runs all six modes once per measurement round. Their order is
randomized identically on every rank so that GPU clock, temperature, and cache
drift do not consistently favor one mode.

It reports:

- `compute_only`: GEMMs without communication.
- `comm_only`: the NCCL collective without GEMMs.
- `serial`: synchronous communication followed by computation.
- `premature_wait`: asynchronous communication followed by an immediate wait.
- `overlap`: independent computation is enqueued before the delayed wait.
- `explicit_stream_overlap`: the same dependency is expressed with a user
  stream and CUDA event.

Each mode reports its median, P10/P90 interval, and coefficient of variation
(`CV = standard deviation / mean`). A low CV means the measurement is stable.
The speedup is calculated as `premature_wait / overlap` (both use `async_op=True`,
so the comparison isolates the effect of scheduling the wait after compute).
A secondary `serial / overlap` comparison is also reported for reference.

For a useful experiment, use at least 30 randomized rounds and tune
`--compute-repeats` and `--comm-size-mb` until
the standalone computation and communication times are reasonably close. The
program prints `workload balance = smaller_time / larger_time`; aim for at
least `0.5`, meaning the two standalone times are within roughly 2x.
The theoretical lower bound for perfect overlap is:

```text
max(compute_time, communication_time)
```

Recommended stability run:

```bash
NUM_GPUS=2 examples/communication_overlap/run.sh \
    --warmup 10 \
    --iterations 100 \
    --comm-size-mb 128 \
    --matrix-size 4096 \
    --compute-repeats 8
```

### Batch Test

`batch_test.sh` automates a sweep over all supported collectives and a range of
communication sizes. It is the easiest way to see how overlap effectiveness
varies with message size without running every combination by hand.

Run with the defaults (2 GPUs, both collectives, 16–256 MB sweep):

```bash
NUM_GPUS=2 bash examples/communication_overlap/batch_test.sh
```

**Configurable via environment variables:**

| Variable | Default | Description |
|---|---|---|
| `NUM_GPUS` | `2` | Number of GPUs |
| `COMM_SIZES_MB` | `16 64 128 256` | Communication sizes to sweep (MB) |
| `MATRIX_SIZE` | `4096` | GEMM dimension |
| `COMPUTE_REPEATS` | `8` | GEMM invocations per measurement |
| `ITERATIONS` | `30` | Measurement rounds (excluding warmup) |
| `WARMUP` | `5` | Warmup rounds |
| `DTYPE` | `bf16` | `bf16` or `fp32` |

**Examples:**

```bash
# 8 GPUs with default sweep
NUM_GPUS=8 bash examples/communication_overlap/batch_test.sh

# Custom sweep range with larger matrices
COMM_SIZES_MB="32 128 512" MATRIX_SIZE=8192 NUM_GPUS=4 \
    bash examples/communication_overlap/batch_test.sh

# Quick smoke test (fewer iterations, fewer sizes)
COMM_SIZES_MB="64 128" ITERATIONS=10 \
    bash examples/communication_overlap/batch_test.sh
```

The script first runs a validation pass (`--validate-only`) to catch
environment problems early, then executes every `(collective, comm_size)`
combination. Results are saved under `batch_results_<timestamp>/`:

```text
batch_results_20260611_143052/
├── validate.log
├── summary.txt                # one-line-per-run table
├── all_reduce_16MB.log
├── all_reduce_64MB.log
├── all_reduce_128MB.log
├── all_reduce_256MB.log
├── all_gather_16MB.log
├── all_gather_64MB.log
├── all_gather_128MB.log
└── all_gather_256MB.log
```

Interpretation:

- Prefer results where `serial/overlap` is consistently above `1.0` across
  P10 through P90.
- Treat a CV above roughly 5% as noisy and investigate GPU sharing, clocks,
  thermal behavior, or background traffic.
- Timing demonstrates an end-to-end benefit; only an Nsight Systems timeline
  proves that NCCL and GEMM kernels overlap.

## HPC-X UCC/UCX Import Errors

An error such as:

```text
libucc.so.1: undefined symbol: ucs_config_doc_nop
```

occurs before this example runs. It means `libucc` was loaded from one HPC-X
installation while `libucs` was loaded from an incompatible UCX installation.
It is not an NCCL overlap error.

`run.sh` detects the standard `/opt/hpcx` layout and places these matching
directories first:

```text
/opt/hpcx/ucx/lib
/opt/hpcx/ucc/lib
```

If HPC-X is installed elsewhere, specify its root:

```bash
HPCX_ROOT=/path/to/hpcx NUM_GPUS=2 \
    examples/communication_overlap/run.sh
```

To inspect the environment manually:

```bash
echo "${LD_LIBRARY_PATH}"
ldd /opt/hpcx/ucc/lib/libucc.so.1 | grep -E 'libucs|libucp|libucm'
```

All reported UCX libraries should come from the same `${HPCX_ROOT}/ucx/lib`
tree. Reinstalling Python packages will not fix this dynamic-library mismatch.

## Verify with Nsight Systems

Timing alone cannot prove overlap. Capture a timeline:

```bash
nsys profile \
    --sample=none \
    --trace=cuda,nvtx,osrt,cublas \
    --output=/tmp/comm_overlap \
    torchrun --standalone --nproc-per-node=2 \
    examples/communication_overlap/overlap_demo.py \
    --collective all_reduce \
    --warmup 2 \
    --iterations 3
```

Open `/tmp/comm_overlap.nsys-rep` in Nsight Systems. On each rank, compare:

- `serial` and `premature_wait`: NCCL and GEMM kernels should be ordered rather
  than substantially concurrent.
- `overlap` and `explicit_stream_overlap`: NCCL kernels should overlap GEMM
  kernels when hardware resources permit.

## Experiments to Try

1. Change `--comm-size-mb` through `16`, `64`, `128`, and `256`.
2. Change `--compute-repeats` until communication is fully or partly hidden.
3. Compare `all_reduce` with `all_gather`.
4. Compare NVLink-connected GPUs with GPUs communicating across PCIe.
5. Compare `CUDA_DEVICE_MAX_CONNECTIONS=1`, `8`, and `32`.

For example:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 NUM_GPUS=2 \
    examples/communication_overlap/run.sh

CUDA_DEVICE_MAX_CONNECTIONS=32 NUM_GPUS=2 \
    examples/communication_overlap/run.sh
```

Do not assume one value is universally best. It changes CUDA work-queue
mapping and can affect launch ordering and concurrency. Megatron configurations
often choose a value deliberately for a specific overlap schedule.

## Connection to Megatron MoE

This experiment is the small-scale version of patterns used in Megatron:

- EP token dispatch runs NCCL communication around expert computation.
- TP communication overlap places all-gather/reduce-scatter near GEMMs.
- Distributed optimizer overlap prefetches parameter all-gathers while prior
  layers are computing.
- Pipeline schedules overlap work from different microbatches when operations
  within one microbatch have unavoidable dependencies.

The production implementation adds buffer lifetime management, process groups,
multiple communication streams, delayed synchronization, and careful scheduling.
The underlying rule remains the same: launch independent operations on different
streams, then synchronize at the latest correct consumption point.
