"""Audit NCCL launch followed by event record on the same stream for rank-0 examples."""
import sqlite3,json,sys
c=sqlite3.connect(sys.argv[1]); c.execute('create index if not exists f_rc on CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)');c.execute('create index if not exists f_ks on CUPTI_ACTIVITY_KIND_KERNEL(globalPid,streamId,start)')
items=json.load(open(sys.argv[2])); out=[]
for z in items:
 if len(z['kernels'])!=1:continue
 k=z['kernels'][0]; cor=k['correlation'];
 rows=c.execute('select k.globalPid from CUPTI_ACTIVITY_KIND_KERNEL k where k.correlationId=? and k.start=?',(cor,k['start_ns'])).fetchall()
 if len(rows)!=1:continue
 pid=rows[0][0]; rt=c.execute('select r.start,r.end from CUPTI_ACTIVITY_KIND_RUNTIME r where (r.globalTid & ~16777215)=? and r.correlationId=?',(pid,cor)).fetchall()
 if len(rt)!=1:continue
 launch=rt[0][0]; candidates=[]
 for event in z['event_records_in_scope']:
  if event['stream']!=k['stream'] or event['record_cpu_ns']<launch:continue
  waits=[w for w in event['waits'] if w['stream']!=k['stream']]
  for w in waits:
   between=c.execute('''select k.correlationId,s.value,k.start,k.end from CUPTI_ACTIVITY_KIND_RUNTIME r join CUPTI_ACTIVITY_KIND_KERNEL k on k.globalPid=(r.globalTid & ~16777215) and k.correlationId=r.correlationId join StringIds s on s.id=k.shortName where k.globalPid=? and k.streamId=? and r.start between ? and ? order by r.start''',(pid,k['stream'],launch,event['record_cpu_ns'])).fetchall()
   nextk=c.execute('''select k.start,k.end,k.streamId,s.value,k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.globalPid=? and k.streamId=? and k.start>=? and lower(s.value) not like '%nccl%' order by k.start limit 1''',(pid,w['stream'],k['end_ns'])).fetchone()
   wait_runtime=c.execute('select r.start,r.end from CUPTI_ACTIVITY_KIND_RUNTIME r where (r.globalTid & ~16777215)=? and r.correlationId=?',(pid,w['correlation'])).fetchall()
   prior=c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.globalPid=? and k.streamId=? and k.end<=? order by k.end desc limit 1',(pid,w['stream'],k['end_ns'])).fetchone()
   next_runtime=c.execute('select r.start,r.end from CUPTI_ACTIVITY_KIND_RUNTIME r where (r.globalTid & ~16777215)=? and r.correlationId=?',(pid,nextk[4])).fetchall() if nextk else []
   candidates.append({'wait_runtime':wait_runtime,'previous_kernel_on_wait_stream':prior,'next_kernel_runtime':next_runtime,'event':event,'wait':w,'launched_from_target_through_record':between,'only_target_launch':len(between)==1 and between[0][0]==cor,'next_non_nccl_kernel_after_target_on_wait_stream':nextk})
 out.append({'metadata':z['metadata'],'nccl_kernel':k,'launch_cpu_ns':launch,'candidates':candidates})
print(json.dumps(out,indent=2))
