import sqlite3,sys,json,collections,bisect
c=sqlite3.connect(sys.argv[1]); q='''select n.text,k.start,k.end,k.deviceId from NVTX_EVENTS n join CUPTI_ACTIVITY_KIND_RUNTIME r on r.globalTid=n.globalTid and r.start between n.start and n.end join CUPTI_ACTIVITY_KIND_KERNEL k on k.globalPid=(r.globalTid & ~16777215) and k.correlationId=r.correlationId join StringIds s on s.id=k.shortName where n.text like 'P17_A2A_%' and s.value like '%nccl%' '''
comm=[r for r in c.execute(q) if not r[0].startswith('P17_A2A_fwd_unknown')]
intervals=collections.defaultdict(list)
for device,start,end in c.execute("select k.deviceId,k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where lower(s.value) not like '%nccl%'"): intervals[device].append((start,end))
merged={}
for d,v in intervals.items():
 out=[]
 for a,b in sorted(v):
  if out and a<=out[-1][1]: out[-1]=(out[-1][0],max(b,out[-1][1]))
  else: out.append((a,b))
 merged[d]=(out,[x[1] for x in out])
stats=collections.defaultdict(lambda:{'count':0,'residence_ns':0,'overlap_ns':0,'overlap_kernel_count':0})
for label,a,b,d in comm:
 k=label.split('_shape_')[0]; x=stats[k]; x['count']+=1; x['residence_ns']+=b-a
 arr,ends=merged[d]; i=bisect.bisect_right(ends,a); overlap=0
 while i<len(arr) and arr[i][0]<b:
  overlap+=max(0,min(b,arr[i][1])-max(a,arr[i][0])); i+=1
 x['overlap_ns']+=overlap; x['overlap_kernel_count']+=overlap>0
for x in stats.values(): x['overlap_fraction']=x['overlap_ns']/x['residence_ns'] if x['residence_ns'] else None
print(json.dumps(stats,indent=2))
