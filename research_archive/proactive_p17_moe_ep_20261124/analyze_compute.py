import sqlite3,json,sys,collections,statistics
c=sqlite3.connect(sys.argv[1]); q='''select n.text,s.value,k.start,k.end from NVTX_EVENTS n join CUPTI_ACTIVITY_KIND_RUNTIME r on r.globalTid=n.globalTid and r.start between n.start and n.end join CUPTI_ACTIVITY_KIND_KERNEL k on k.globalPid=(r.globalTid & ~16777215) and k.correlationId=r.correlationId join StringIds s on s.id=k.shortName where n.text in ('P17_experts','P17_router','P17_checkpoint_replay')'''
g=collections.defaultdict(list)
for label,name,start,end in c.execute(q):
 kind='gemm' if 'gemm' in name.lower() else 'other'
 g[label+':'+kind].append((end-start)/1000)
out={k:{'kernel_count':len(v),'mean_us':statistics.mean(v),'median_us':statistics.median(v),'total_ms':sum(v)/1000} for k,v in g.items()}
out['nvtx_range_counts']={k:v for k,v in c.execute("select text,count(*) from NVTX_EVENTS where text in ('P17_experts','P17_router','P17_checkpoint_replay','P17_checkpoint_forward') group by text")}
print(json.dumps(out,indent=2))
