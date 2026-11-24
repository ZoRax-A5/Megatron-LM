"""Conservatively link the captured TP2 sequence to NVTX and CUDA event records."""
import argparse, collections, json, sqlite3
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('case',type=Path); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
c=sqlite3.connect(args.case/'trace.sqlite'); names=dict(c.execute('select id,value from StringIds'))
api=collections.defaultdict(list)
for a,b,tid,cor in c.execute('select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'):
 api[(tid>>24,cor)].append((a,b,tid))
records=collections.defaultdict(list)
for gp,e,s,st,cor in c.execute('select globalPid,eventId,eventSyncId,streamId,correlationId from CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
 pid=gp>>24
 for a,b,tid in api[pid,cor]: records[pid,e,s].append(dict(stream=st,start=a,end=b,corr=cor))
waits=collections.defaultdict(list)
for gp,e,s,st,cor in c.execute('select globalPid,eventId,eventSyncId,streamId,correlationId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2'):
 pid=gp>>24
 for a,b,tid in api[pid,cor]: waits[pid,e,s].append(dict(stream=st,start=a,end=b,corr=cor))
result=[]
for rank in (0,1):
 xs=json.loads((args.case/f'tensor_rank{rank}.json').read_text()); ar=[x for x in xs if x['kind']=='all_reduce']; q=[x for x in xs if x['kind']=='qkv_input']; x=ar[0]
 assert x['sequence']==1119 and x['peers']==[0,1] and len(q)==1
 scopes=[]
 for a,b,txt,tid in c.execute("select start,end,text,globalTid from NVTX_EVENTS where text like '%all_reduce_api%'"):
  try: m=json.loads(txt.split('|',1)[1])
  except (ValueError,IndexError): continue
  if m.get('rank')==rank and m.get('sequence')==x['sequence'] and m.get('stage')==['backward',7]: scopes.append((a,b,tid))
 assert len(scopes)==1
 a,b,tid=scopes[0]; pid=tid>>24
 ks=[]
 for ts,cor,ga,gb,stream,name in c.execute('select r.start,r.correlationId,k.start,k.end,k.streamId,k.shortName from CUPTI_ACTIVITY_KIND_RUNTIME r join CUPTI_ACTIVITY_KIND_KERNEL k on k.globalPid=(r.globalTid & ~16777215) and k.correlationId=r.correlationId where r.globalTid=? and r.start between ? and ?', (tid,a,b)):
  if 'allreduce' in names[name].lower().replace('_',''): ks.append(dict(submit=ts,corr=cor,gpu_start=ga,gpu_end=gb,stream=stream,name=names[name]))
 assert len(ks)==1
 k=ks[0]; ev=[]
 for key,rs in records.items():
  if key[0]!=pid or len(rs)!=1:continue
  rec=rs[0]
  if rec['stream']!=k['stream'] or not(k['submit']<rec['start']<b):continue
  ws=[w for w in waits[key] if rec['end']<=w['start']<b and w['stream']!=rec['stream']]
  for w in ws:
   prior=[r['start'] for (pp,ee,ss),rr in records.items() if pp==pid for r in rr if r['stream']==rec['stream'] and r['start']<rec['start']]
   previous=max(prior,default=-1)
   launches=[]
   for (pp,cor),calls in api.items():
    if pp!=pid or len(calls)!=1 or not previous<calls[0][0]<rec['start']:continue
    rows=c.execute('select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and correlationId=? and streamId=?',(pid<<24,cor,rec['stream'])).fetchall()
    launches.extend(dict(corr=cor,name=names[n],gpu_start=ga,gpu_end=gb) for ga,gb,n in rows)
   ev.append(dict(event=key[1],sync=key[2],record=rec,wait=w,launches_since_previous_record=launches,unique_target_producer=len(launches)==1 and launches[0]['corr']==k['corr']))
 # These are candidate associations; without an intervening-kernel audit they are not proven producer edges.
 result.append(dict(rank=rank,tp_collective=x,qkv=q[0],scope=dict(start=a,end=b),kernel=k,event_candidates=ev,unknown=['actual tensor consumer identity','QKV replay input readiness/alias at backward trigger','legal insertion boundary','critical path exposure']))
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{'rank':r['rank'],'kernel_ms':(r['kernel']['gpu_end']-r['kernel']['gpu_start'])/1e6,'event_candidates':r['event_candidates']} for r in result],indent=2))
