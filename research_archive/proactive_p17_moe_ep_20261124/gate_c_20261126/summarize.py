import json,re,sys,glob,statistics,collections
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for p in sorted(root.glob('*.log')):
 if not re.fullmatch(r'(C0|Cnoop|C1|C2|C1serial)_r[123]\.log',p.name):continue
 text=p.read_text(errors='replace');matches=re.findall(r'iteration\s+(\d+)/\s*50.*?elapsed time per iteration \(ms\):\s*([\d.]+).*?lm loss:\s*([\d.E+-]+).*?grad norm:\s*([\d.E+-]+).*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)',text)
 stable=[float(m[1]) for m in matches if 21<=int(m[0])<=50];peaks=[];life=[]
 for f in glob.glob(str(p.with_suffix(''))+'.rank*.jsonl'):
  peaks.extend(x for line in open(f) if 'peak_allocated' in (x:=json.loads(line)))
 for f in glob.glob(str(p.with_suffix(''))+'.lifecycle.rank*.jsonl'):
  life.extend(json.loads(line) for line in open(f))
 events=collections.Counter(e['event'] for r in life for e in r['events'])
 lifetimes=[e['cache_lifetime_seconds'] for r in life for e in r['events'] if e['event']=='released' and e.get('cache_lifetime_seconds',0)>0]
 cachebytes=[e['output_bytes'] for r in life for e in r['events'] if e['event']=='prefetch_complete']
 rows.append({'file':p.name,'steps':len(matches),'steady_ms_mean':statistics.mean(stable) if len(stable)==30 else None,'steady_ms_std':statistics.stdev(stable) if len(stable)==30 else None,'tokens_s':65536/(statistics.mean(stable)/1000) if len(stable)==30 else None,'max_allocated':max((x['peak_allocated'] for x in peaks),default=None),'max_reserved':max((x['peak_reserved'] for x in peaks),default=None),'skipped':sum(int(x[4]) for x in matches),'nan':sum(int(x[5]) for x in matches),'final_loss':matches[-1][2] if matches else None,'final_grad_norm':matches[-1][3] if matches else None,'lifecycle_instances':len(life),'lifecycle_events':dict(events),'all_graphs_released':all(r.get('graph_released',False) and r['cache_bytes']==0 for r in life),'cache_output_bytes_range':[min(cachebytes),max(cachebytes)] if cachebytes else None,'cpu_cache_lifetime_ms_median':statistics.median(lifetimes)*1000 if lifetimes else None})
print(json.dumps(rows,indent=2))
