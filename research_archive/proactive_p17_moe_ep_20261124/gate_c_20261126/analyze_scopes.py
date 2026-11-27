"""Non-overlapping wall-time partitions of target CPU scopes; no hardware CPU attribution."""
import sqlite3,json,sys,collections,statistics
c=sqlite3.connect(sys.argv[1]);c.execute('create index if not exists scope_rt on CUPTI_ACTIVITY_KIND_RUNTIME(globalTid,start)');c.execute('create index if not exists scope_nv on NVTX_EVENTS(globalTid,start)')
def union(values):
 out=[]
 for a,b in sorted(values):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
  else:out.append((a,b))
 return out
def size(v):return sum(b-a for a,b in union(v))
def overlap(v,w):return sum(max(0,min(b,d)-max(a,e)) for a,b in union(v) for e,d in union(w))
rows=[]
for step in (20,21,22):
 for kind,label,child in [('prefetch',f'proactive_recompute.prefetch.{step}',f'P17_FC1_CALL_prefetch_{step}'),('active_backward',f'P17_ACTIVE_BACKWARD_{step}',f'P17_FC1_CALL_backward_{step}'),('native_backward',f'P17_NATIVE_BACKWARD_{step}','P17_checkpoint_replay')]:
  scopes=list(c.execute('select start,end,globalTid from NVTX_EVENTS where text=?',(label,)))
  for a,b,tid in scopes:
   children=list(c.execute('select start,end from NVTX_EVENTS INDEXED BY scope_nv where globalTid=? and start between ? and ? and text=?',(tid,a,b,child)))
   children=[(max(a,x),min(b,y)) for x,y in children if y is not None]
   apis=list(c.execute('select r.start,r.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r INDEXED BY scope_rt join StringIds s on s.id=r.nameId where r.globalTid=? and r.start between ? and ?',(tid,a,b)))
   api_intervals=[(max(a,x),min(b,y)) for x,y,n in apis];child_ns=size(children);outside_api_ns=size(api_intervals)-overlap(api_intervals,children)
   rows.append({'kind':kind,'step':step,'pid':tid & ~16777215,'parent_us':(b-a)/1000,'child_forward_us':child_ns/1000,'cuda_api_outside_child_us':outside_api_ns/1000,'remaining_wall_us':((b-a)-child_ns-outside_api_ns)/1000,'api_us_by_name':{n:sum(max(0,min(b,y)-max(a,x)) for x,y,k in apis if k==n)/1000 for n in sorted({n for x,y,n in apis})}})
summary={}
for kind in sorted({r['kind'] for r in rows}):
 rr=[r for r in rows if r['kind']==kind];summary[kind]={'samples':len(rr),**{key:statistics.mean(r[key] for r in rr) for key in ('parent_us','child_forward_us','cuda_api_outside_child_us','remaining_wall_us')}}
print(json.dumps({'summary':summary,'rows':rows},indent=2))
