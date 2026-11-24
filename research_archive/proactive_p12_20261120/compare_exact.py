"""Compare all participating ranks, without silently treating incomplete runs as passes."""
import json
from pathlib import Path
import sys

reference = Path(sys.argv[1])
paths = [Path(v) for v in sys.argv[2:]]
cases = {}
for case in [reference, *paths]:
    failures = []
    for rank in range(8):
        a = json.loads((reference/f"exact_rank{rank}.json").read_text())
        b = json.loads((case/f"exact_rank{rank}.json").read_text())
        assert len(a) == len(b) == 3
        for prev, now in zip(a,b):
            assert prev.keys() == now.keys()
            if prev != now:
                failures.append(dict(rank=rank,iteration=prev["iteration"],
                                     fields=[key for key in prev if prev[key]!=now[key]]))
    cases[case.name] = dict(checks=8*3*2, mismatches=failures)
print(json.dumps(cases,indent=2))
assert not any(x["mismatches"] for x in cases.values())
