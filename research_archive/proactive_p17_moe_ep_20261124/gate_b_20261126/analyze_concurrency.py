import sqlite3,json,sys,collections,statistics
c=sqlite3.connect(sys.argv[1]);c.execute('create index if not exists p17rt on CUPTI_ACTIVITY_KIND_RUNTIME(globalTid,start)');c.execute('create index if not exists p17k on CUPTI_ACTIVITY_KIND_KERNEL(globalPid,correlationId)')
rows={}
for kind in ('combine','replay'):
 for step in (20,21,22):
  label=f'P17_TARGET_COMBINE_{step}' if kind=='combine' else f'proactive_recompute.prefetch.{step}'
  q='''select k.globalPid,k.start,k.end,k.streamId,s.value,r.start,r.end from NVTX_EVENTS n join CUPTI_ACTIVITY_KIND_RUNTIME r on r.globalTid=n.globalTid and r.start between n.start and n.end join CUPTI_ACTIVITY_KIND_KERNEL k on k.globalPid=(r.globalTid & ~16777215) and k.correlationId=r.correlationId join StringIds s on s.id=k.shortName where n.text=?'''
  for pid,a,b,stream,name,ra,rb in c.execute(q,(label,)):
   if kind=='combine' and 'nccl' not in name.lower():continue
   rows.setdefault((pid,step),{'pid':pid,'step':step,'combine':[],'replay':[]})[kind].append({'start':a,'end':b,'stream':stream,'name':name,'cpu_submit_start':ra,'cpu_submit_end':rb})
def union(kernels):
 merged=[]
 for a,b in sorted((k['start'],k['end']) for k in kernels):
  if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
  else:merged.append((a,b))
 return merged
out=[]
for r in rows.values():
 if not r['combine'] or not r['replay']:continue
 comm,replay=union(r['combine']),union(r['replay'])
 gemm=union([k for k in r['replay'] if 'gemm' in k['name'].lower()])
 r['gemm_us']=sum(b-a for a,b in gemm)/1000
 r['gemm_overlap_us']=sum(max(0,min(b,d)-max(a,c)) for a,b in comm for c,d in gemm)/1000
 overlap=sum(max(0,min(b,d)-max(a,c)) for a,b in comm for c,d in replay)
 r.update(overlap_us=overlap/1000,combine_us=sum(b-a for a,b in comm)/1000,replay_us=sum(b-a for a,b in replay)/1000)
 out.append(r)
summary={'samples':len(out),'overlapped_samples':sum(r['overlap_us']>0 for r in out),'overlapped_gemm_samples':sum(r['gemm_overlap_us']>0 for r in out)}
for key in ('combine_us','replay_us','overlap_us','gemm_us','gemm_overlap_us'):
 v=[r[key] for r in out];summary[key]={'mean':statistics.mean(v),'median':statistics.median(v),'min':min(v),'max':max(v)} if v else None
print(json.dumps({'summary':summary,'samples':out},indent=2))
