"""Audit stream-event candidates; matching event id alone is not a producer proof."""
import sqlite3,json,sys
c=sqlite3.connect(sys.argv[1]);c.execute('create index if not exists f_rt on CUPTI_ACTIVITY_KIND_RUNTIME(globalTid,start)');c.execute('create index if not exists f_ev on CUPTI_ACTIVITY_KIND_CUDA_EVENT(globalPid,correlationId)');c.execute('create index if not exists f_wait on CUPTI_ACTIVITY_KIND_SYNCHRONIZATION(globalPid,eventId,eventSyncId)');c.execute('create index if not exists f_k on CUPTI_ACTIVITY_KIND_KERNEL(globalPid,correlationId)')
selected=[]
for label,a,b,tid in c.execute("select text,start,end,globalTid from NVTX_EVENTS where text like 'P17META_%'"):
 m=json.loads(label[len('P17META_'):])
 if m.get('rank')!=0 or m.get('step')!=21 or m.get('layer')!=3 or m.get('microbatch')!=1:continue
 if (m.get('op'),m.get('phase')) not in [('token_dispatch','forward'),('token_combine','forward'),('token_combine','backward')]:continue
 if m['op']=='token_dispatch' and len(m['shape'])!=2:continue
 selected.append((m,a,b,tid))
out=[]
for m,a,b,tid in selected:
 pid=tid & ~16777215
 api=c.execute('select start,end,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start between ? and ?', (tid,a,b)).fetchall()
 kernels=[];records=[]
 for start,end,cor,name in api:
  for ka,kb,stream,kname in c.execute('select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.globalPid=? and k.correlationId=?',(pid,cor)):
   if 'nccl' in kname.lower():kernels.append({'start_ns':ka,'end_ns':kb,'stream':stream,'kernel':kname,'correlation':cor})
  for eid,sync,stream in c.execute('select eventId,eventSyncId,streamId from CUPTI_ACTIVITY_KIND_CUDA_EVENT where globalPid=? and correlationId=?',(pid,cor)):
   wait=[{'start_ns':x,'end_ns':y,'stream':st,'correlation':cc} for x,y,st,cc in c.execute('select start,end,streamId,correlationId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where globalPid=? and eventId=? and eventSyncId=? and syncType=2',(pid,eid,sync))]
   records.append({'event':eid,'generation':sync,'record_cpu_ns':start,'stream':stream,'waits':wait[:20]})
 out.append({'metadata':m,'cpu_range_ns':[a,b],'kernels':kernels,'event_records_in_scope':records})
print(json.dumps(out,indent=2))
