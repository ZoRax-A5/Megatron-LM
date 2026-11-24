import sqlite3,sys,json,statistics,collections
c=sqlite3.connect(sys.argv[1]); c.execute('create index if not exists p17_rt on CUPTI_ACTIVITY_KIND_RUNTIME(globalTid,start)'); c.execute('create index if not exists p17_k on CUPTI_ACTIVITY_KIND_KERNEL(globalPid,correlationId)')
q='''select n.text,k.start,k.end,k.deviceId,k.streamId,s.value from NVTX_EVENTS n join CUPTI_ACTIVITY_KIND_RUNTIME r on r.globalTid=n.globalTid and r.start between n.start and n.end join CUPTI_ACTIVITY_KIND_KERNEL k on k.globalPid=(r.globalTid & ~16777215) and k.correlationId=r.correlationId join StringIds s on s.id=k.shortName where n.text like 'P17_A2A_%' and s.value like '%nccl%' '''
rows=list(c.execute(q)); groups=collections.defaultdict(list)
for label,start,end,device,stream,name in rows:
 if label.startswith('P17_A2A_fwd_unknown'): continue
 kind=label.split('_shape_')[0]
 if kind=='P17_A2A_fwd_P17_dispatch': kind += '_tokens' if ', 512)' in label else '_probs'
 groups[kind].append((end-start)/1000)
out={k:{'kernel_count':len(v),'mean_us':statistics.mean(v),'median_us':statistics.median(v),'min_us':min(v),'max_us':max(v)} for k,v in groups.items()}
out['kernel_names']=dict(collections.Counter(r[-1] for r in rows if not r[0].startswith('P17_A2A_fwd_unknown')))
out['note']='GPU kernel residence per matched runtime launch, all ranks/3 profiler steps; not independent cost or exposed wait.'
print(json.dumps(out,indent=2))
