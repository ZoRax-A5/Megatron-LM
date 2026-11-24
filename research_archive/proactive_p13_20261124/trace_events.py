import sqlite3,json,sys,statistics,collections
from pathlib import Path
p=Path(sys.argv[1]); c=sqlite3.connect(p/'trace.sqlite')
runtime={(tid>>24,cor):(a,b,tid) for a,b,tid,cor in c.execute('select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
records={}
for pid,eid,sid,stream,corr in c.execute('select globalPid,eventId,eventSyncId,streamId,correlationId from CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
 pid>>=24
 if (pid,corr) in runtime:records[pid,eid,sid]=(stream,*runtime[pid,corr][:2])
waits=collections.defaultdict(list)
for pid,eid,sid,stream,corr in c.execute('select globalPid,eventId,eventSyncId,streamId,correlationId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2'):
 pid>>=24
 if (pid,corr) in runtime and (pid,eid,sid) in records:
  src,a,b=records[pid,eid,sid];w,z,t=runtime[pid,corr];waits[pid,src].append((a,w,stream,eid,sid))
for v in waits.values():v.sort()
rows=[];comm=json.loads((p/'comm.json').read_text()); marks=json.loads((p/'nvtx_attribution.json').read_text())
import bisect
for rank,marks_r in marks.items():
 for e in marks_r:
  if e['kind']!='all_reduce_api' or len(e.get('peers',[])) not in (2,4):continue
  for k in e['kernels']:
   if 'allreduce' not in k['name'].lower().replace('_',''):continue
   t=runtime.get((k['pid'],k['correlation']))
   if not t:continue
   w=waits[k['pid'],k['stream']];i=bisect.bisect_left(w,(t[0],-1))
   options=[x for x in w[i:i+6] if x[0]<e['cpu_end'] and x[1]<e['cpu_end'] and x[2]!=k['stream']]
   if options:
    a,b,s,eid,sid=options[0];rows.append({'rank':int(rank),'stage':e['stage'],'peers':e['peers'],'nccl_stream':k['stream'],'wait_stream':s,'record_submit_ns':a,'wait_submit_ns':b,'kernel_start':k['start'],'kernel_end':k['end'],'event':eid,'event_sync':sid,'bounded_by_allreduce_api':True})
(p/'event_wait_attribution.json').write_text(json.dumps(rows,indent=2))
print('count',len(rows),'stream pairs',collections.Counter((x['nccl_stream'],x['wait_stream']) for x in rows).most_common(10),'stages',collections.Counter(x['stage'][0] for x in rows))
