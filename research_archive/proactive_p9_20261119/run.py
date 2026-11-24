"""Run matched Qwen3 training cases and gate progression on exact comparisons."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

here = Path(__file__).resolve().parent
root = here.parents[1]
repo = root.parent / "Megatron-LM-layerwise"
sys.path.insert(0, str(root))
import bench

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=("single", "tp2", "dp4", "perf", "timeline"))
parser.add_argument("--tag", default="")
args = parser.parse_args()
gpus, tp, counts = {"single": (1, 1, (2, 4, 8)), "tp2": (2, 2, (4,)),
                    "dp4": (8, 2, (8,)), "perf": (8, 2, (8,)),
                    "timeline": (8, 2, (8,))}[args.phase]
verify = args.phase in ("single", "tp2", "dp4")
config = dict(num_hidden_layers=2, hidden_size=256, intermediate_size=512,
              num_attention_heads=4, num_key_value_heads=2, head_dim=64,
              vocab_size=2048, rms_norm_eps=1e-6, rope_theta=1000000,
              initializer_range=0.02, max_position_embeddings=512)
for n in counts:
    suite = here / f"{args.phase}_mb{n}{args.tag}"
    suite.mkdir()
    baseline = None
    full_losses = {}
    sync_baseline = {}
    modes = ("S0", "S0repeat", "S1", "S2") if verify else ("S0", "S1", "S2")
    for mode in modes:
        case = suite / mode
        case.mkdir()
        steps = 30 if args.phase == "perf" else 3
        options = SimpleNamespace(gpus=gpus, megatron=repo, seed=1234, iters=steps,
                                  seq_length=256, micro_batch=1, global_batch=n*gpus//tp,
                                  sequence_parallel=False, overlap=False)
        cmd = bench.command(options, config, tp, 1)
        cmd[cmd.index(str(repo / "pretrain_gpt.py"))] = str(here / "entry.py")
        cmd += ["--recompute-granularity", "selective", "--recompute-layer-config",
                '{"0":["attn_qkv"],"1":["attn_qkv"]}', "--overlap-grad-reduce",
                "--ddp-bucket-size", "400000"]
        if mode in ("S1", "S2"):
            cmd += ["--cross-microbatch-recompute-schedule"]
        if mode == "S2":
            cmd += ["--proactive-recompute", "--proactive-recompute-max-cached-checkpoints", "2"]
        settings = dict(QWEN_BENCH_MEGATRON=str(repo), QWEN_BENCH_CASE=str(case),
                        P1_STRATEGY="B1" if mode == "S0" else mode,
                        P1_REFERENCE=str(suite / "S0"), P9_PHASE=args.phase,
                        CUDA_DEVICE_MAX_CONNECTIONS="1", OMP_NUM_THREADS="2",
                        NVTE_ALLOW_NONDETERMINISTIC_ALGO="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        (case / "manifest.json").write_text(json.dumps(dict(cmd=cmd, env=settings), indent=2))
        with (case / "train.log").open("w") as log:
            result = subprocess.run(cmd, cwd=repo, env={**os.environ, **settings},
                                    stdout=log, stderr=subprocess.STDOUT, timeout=420)
        result.check_returncode()
        lines = (case / "train.log").read_text().splitlines()
        records = [line for line in lines if "elapsed time per iteration" in line]
        losses = [re.search(r"lm loss: (\S+)", line)[1] for line in records]
        assert len(losses) == steps, (case, losses)
        assert all(re.search(r"skipped iterations:\s+0.*nan iterations:\s+0", line)
                   for line in records), case
        if baseline is None:
            baseline = losses
        assert losses == baseline, (case, losses, baseline)
        for rank in range(gpus):
            data = json.loads((case / f"rank{rank}.json").read_text())
            assert len(data["steps"]) == steps
            expected = n * 2
            for step in data["steps"]:
                assert step["qkv_calls"] == 2 * expected, (case, rank, step)
                key = (rank, step["iteration"])
                if "losses" in step:
                    assert full_losses.setdefault(key, step["losses"]) == step["losses"]
                dispatch = [(row["stage"], row["last"], row["async_enabled"]) for row in step["sync"]]
                assert all(last and enabled and stage == ["backward", n-1]
                           for stage, last, enabled in dispatch), (case, rank, dispatch)
                assert dispatch and sync_baseline.setdefault(key, dispatch) == dispatch
                if gpus == 8:
                    assert len(step["reduce_scatter"]) == len(dispatch)
                    assert all(row["async_op"] and row["stage"] == ["backward", n-1]
                               for row in step["reduce_scatter"])
                calls = step["calls"]
                targets = set(range(0, n if gpus == tp else n-2, 2)) if mode == "S2" else set()
                for mb in range(n):
                    assert calls.get(str(("forward", mb))) == 2
                    assert calls.get(str(("prefetch", mb)), 0) == (2 if mb in targets else 0)
                    assert calls.get(str(("backward", mb)), 0) == (0 if mb in targets else 2)
                if mode == "S2":
                    hits = 2 * ((n if gpus == tp else n-1)//2)
                    assert step["hits"] == hits and step["released"] == expected, step
                    assert all(row["final_state"] == "consumed" for row in step["checkpoints"])
                    assert step["max_cached"] == min(2, hits)
                    for row in step["checkpoints"]:
                        names = [event["event"] for event in row["events"]]
                        assert names.count("released") == 1
                        if row["hit"]:
                            assert names.count("cache_consumed") == 1 and "fallback" not in names
                            assert names.count("prefetch_complete") == 1
                        else:
                            assert names.count("fallback") == 1
                    if args.phase == "timeline":
                        gpu = {name: elapsed for name, _, elapsed in step["timeline"]}
                        for row in step["checkpoints"]:
                            if row["hit"]:
                                mb = row["metadata"]["microbatch"]
                                cid = row["id"]
                                key = f"replay:{cid}:{mb+1}"
                                assert (gpu[f"forward:{mb}:end"] < gpu[f"forward:{mb+1}:start"]
                                        < gpu[key+":start"] < gpu[key+":prefetch_start"]
                                        < gpu[key+":prefetch_complete"] < gpu[key+":end"]
                                        < gpu[f"forward:{mb+1}:end"] < gpu[f"backward:{mb}:start"]
                                        < gpu[f"consume:{cid}"] < gpu[f"release:{cid}"])

            if verify:
                checked = json.loads((case / f"verification_rank{rank}.json").read_text())
                assert checked["completed"]
                if mode != "S0":
                    assert len(checked["checks"]) == 2 * steps, checked
                    assert all(c["max_abs"] == 0 and c["bad_elements"] == 0
                               for c in checked["checks"]), (case, rank, checked["checks"])
        print(f"PASS {case}", flush=True)
