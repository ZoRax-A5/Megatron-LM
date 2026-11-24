import json,sqlite3,statistics,sys
from pathlib import Path
p=Path(sys.argv[1]); c=sqlite3.connect(p/'trace.sqlite'); ids=dict(c.execute('select id,value from StringIds'))
steps=[]
for a,b,t,tid in c.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'P10|%' and end is not null"):
 d=json.loads(t[4:])
 if d['kind']=='step':steps.append((d['rank'],tid>>24,a,b))
def union(xs):
 z=[]
 for a,b in sorted(xs):
  if z and a<=z[-1][1]:z[-1][1]=max(z[-1][1],b)
  else:z.append([a,b])
 return sum(b-a for a,b in z)/1e6
out=[]
for rank,pid,start,end in steps:
 ks=[(a,b,ids[n]) for a,b,n in c.execute('select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and start>=? and start<?',(pid<<24,start,end))]
 def get(s):return [(a,b) for a,b,n in ks if s in n.lower().replace('_','')]
 ar=get('allreduce'); rs=get('reducescatter'); ag=get('allgather'); nccl=[(a,b) for a,b,n in ks if 'nccl' in n.lower()]; compute=[(a,b) for a,b,n in ks if 'nccl' not in n.lower()]
 out.append(dict(rank=rank,step_cpu_ms=(end-start)/1e6,gpu_span_ms=(max(b for a,b,n in ks)-min(a for a,b,n in ks))/1e6,allreduce_count=len(ar),allreduce_union_ms=union(ar),allreduce_kernel_median_ms=statistics.median((b-a)/1e6 for a,b in ar) if ar else None,rs_count=len(rs),rs_union_ms=union(rs),ag_count=len(ag),ag_union_ms=union(ag),nccl_union_ms=union(nccl),compute_union_ms=union(compute)))
(p/'aggregate.json').write_text(json.dumps(out,indent=2));print(p.name,len(out));print(out)
