import glob, json, re, statistics, sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for p in sorted(root.glob('ep*_*.log')):
    if not re.fullmatch(r'ep(?:2_run1|[248]_measure[123]|2_fc1_measure[123])\.log',p.name): continue
    s=p.read_text(errors='replace')
    found=re.findall(r'iteration\s+(\d+)/\s*50.*?elapsed time per iteration \(ms\):\s*([\d.]+).*?lm loss:\s*([\d.E+-]+).*?grad norm:\s*([\d.E+-]+).*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)',s)
    stable=[float(x[1]) for x in found if 21 <= int(x[0]) <= 50]
    mem=[]
    for f in glob.glob(str(p.with_suffix(''))+'.rank*.jsonl'):
        for line in open(f):
            x=json.loads(line)
            if 'peak_allocated' in x: mem.append(x)
    rows.append({'file':p.name,'iterations':len(found),'stable_ms_mean':statistics.mean(stable) if len(stable)==30 else None,'stable_ms_std':statistics.stdev(stable) if len(stable)==30 else None,'tokens_per_s':65536/(statistics.mean(stable)/1000) if len(stable)==30 else None,'last_loss':float(found[-1][2]) if found else None,'last_grad_norm':float(found[-1][3]) if found else None,'skipped_total':sum(int(x[4]) for x in found),'nan_total':sum(int(x[5]) for x in found),'peak_allocated_bytes_max_rank':max((x['peak_allocated'] for x in mem),default=None),'peak_reserved_bytes_max_rank':max((x['peak_reserved'] for x in mem),default=None)})
print(json.dumps(rows,indent=2))
