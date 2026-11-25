import sys,json,re,glob,statistics
from pathlib import Path
root=Path(sys.argv[1]);out={}
for mode in ('baseline','fc1'):
 rows=[]
 for run in range(1,4):
  stem=root/f'large_{mode}_{run}'; p=stem.with_suffix('.log')
  if not p.exists():continue
  s=p.read_text(errors='replace'); matches=re.findall(r'iteration\s+(\d+)/\s*30.*?elapsed time per iteration \(ms\):\s*([\d.]+).*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)',s)
  vals=[float(t) for i,t,skip,nan in matches if 11<=int(i)<=30]; peaks=[]
  for f in glob.glob(str(stem)+'.rank*.jsonl'):
   peaks.extend(d for line in open(f) if 'peak_allocated' in (d:=json.loads(line)))
  rows.append({'file':str(p.name),'iterations':len(matches),'steady_mean_ms':statistics.mean(vals) if len(vals)==20 else None,'steady_sd_ms':statistics.stdev(vals) if len(vals)==20 else None,'peak_allocated_bytes':max((d['peak_allocated'] for d in peaks),default=None),'peak_reserved_bytes':max((d['peak_reserved'] for d in peaks),default=None),'skipped':sum(int(x[2]) for x in matches),'nan':sum(int(x[3]) for x in matches)})
 out[mode]=rows
print(json.dumps(out,indent=2))
