"""Conservative audit of event lifetimes. Ambiguity remains explicit."""
import collections,json,sqlite3,sys
from pathlib import Path
p=Path(sys.argv[1]); c=sqlite3.connect(p/'trace.sqlite')
api=collections.defaultdict(list)
for a,b,tid,cor in c.execute('select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'):
 api[(tid>>24,cor)].append((a,b,tid))
events=collections.defaultdict(list)
for pid,eid,sid,stream,cor in c.execute('select globalPid,eventId,eventSyncId,streamId,correlationId from CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
 pid>>=24
 for a,b,tid in api[pid,cor]: events[(pid,eid,sid)].append(dict(stream=stream,start=a,end=b,tid=tid,corr=cor))
# Events with no runtime timestamp cannot be assigned a lifetime.
raw=collections.Counter((pid>>24,eid,sid) for pid,eid,sid in c.execute('select globalPid,eventId,eventSyncId from CUPTI_ACTIVITY_KIND_CUDA_EVENT'))
counts=collections.Counter(); waits=[]
for pid,eid,sid,stream,cor in c.execute('select globalPid,eventId,eventSyncId,streamId,correlationId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2'):
 pid>>=24; key=(pid,eid,sid); w=api[pid,cor]; ev=events[key]
 if len(ev)!=1 or len(w)!=1:
  counts['unmatched_or_ambiguous']+=1;continue
 rec=ev[0]; a,b,tid=w[0]
 if rec['end']>a:
  counts['host_order_unknown']+=1;continue
 counts['matched_unique_host_order']+=1
 waits.append(dict(pid=pid,event=eid,sync=sid,record=rec,wait_stream=stream,wait_start=a,wait_end=b,wait_tid=tid,wait_corr=cor))
counts['record_keys_duplicate']=sum(v>1 for v in raw.values())
counts['record_api_missing_or_ambiguous']=sum(len(events[k])!=1 for k in raw)
# Associate kernel only when exactly one NCCL launch from its stream is enclosed
# between prior record on that stream and this record, by host API submit order.
streams=collections.defaultdict(list)
for v in events.values():
 if len(v)==1:streams[(v[0]['tid']>>24,v[0]['stream'])].append(v[0]['start'])
for x in streams.values():x.sort()
import bisect
kernels=collections.defaultdict(list)
ids=dict(c.execute('select id,value from StringIds'))
for a,b,pid,stream,cor,name in c.execute('select start,end,globalPid,streamId,correlationId,shortName from CUPTI_ACTIVITY_KIND_KERNEL'):
 pid>>=24
 if (pid,cor) in api and len(api[pid,cor])==1:
  kernels[pid,stream].append((api[pid,cor][0][0],a,b,cor,ids[name]))
for v in kernels.values():v.sort()
associated=[]
for w in waits:
 rec=w['record']; key=(w['pid'],rec['stream']); seq=streams[key]; i=bisect.bisect_left(seq,rec['start']); prev=seq[i-1] if i else -1
 ks=kernels[key]; lo=bisect.bisect_right(ks,(prev,)); hi=bisect.bisect_left(ks,(rec['start'],)); between=ks[lo:hi]
 if len(between)!=1:counts['producer_kernel_unknown']+=1;continue
 kernel=between[0]
 counts['unique_producer_kernel']+=1
 if 'nccl' in kernel[-1].lower():
  counts['unique_nccl_producer']+=1
  associated.append(dict(wait=w,kernel=dict(submit=kernel[0],gpu_start=kernel[1],gpu_end=kernel[2],corr=kernel[3],name=kernel[4])))
out=dict(case=p.name,counts=dict(counts),matched_nccl=associated)
(Path(__file__).parent/(p.name+'_event_audit.json')).write_text(json.dumps(out,indent=2));print(p.name,counts)
