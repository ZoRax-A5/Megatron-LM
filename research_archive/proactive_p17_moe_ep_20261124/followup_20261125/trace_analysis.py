"""Map identity ranges to launched NCCL GPU kernels in warmed steps 20..22 (zero based)."""
import sqlite3,json,sys,statistics,collections,os
c=sqlite3.connect(sys.argv[1]); c.execute('create index if not exists p17rt on CUPTI_ACTIVITY_KIND_RUNTIME(globalTid,start)');c.execute('create index if not exists p17kg on CUPTI_ACTIVITY_KIND_KERNEL(globalPid,correlationId)');c.execute('create index if not exists p17ks on CUPTI_ACTIVITY_KIND_KERNEL(globalPid,streamId,start)')
q='''select n.text,n.start,n.end,k.globalPid,k.start,k.end,k.streamId,s.value from NVTX_EVENTS n join CUPTI_ACTIVITY_KIND_RUNTIME r on r.globalTid=n.globalTid and r.start between n.start and n.end join CUPTI_ACTIVITY_KIND_KERNEL k on k.globalPid=(r.globalTid & ~16777215) and k.correlationId=r.correlationId join StringIds s on s.id=k.shortName where n.text like 'P17META_%' and s.value like '%nccl%' '''
steps=set(map(int,os.getenv('P17_STEPS','20,21,22').split(',')))
rows=[]
for label,ns,ne,pid,start,end,stream,kernel in list(c.execute(q)):
 meta=json.loads(label[len('P17META_'):]);
 if meta.get('step') not in steps or not meta.get('op'):continue
 if meta['phase']=='backward':
  # The backward NVTX scope carries its forward-origin metadata; autograd swaps splits.
  meta['forward_origin_shape']=meta['shape']
  ins,outs=meta.get('in_splits'),meta.get('out_splits')
  if ins is not None and outs is not None:
   meta['in_splits'],meta['out_splits']=outs,ins
   meta['shape']=[sum(outs)]+meta['shape'][1:]
 nextk=c.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.globalPid=? and k.streamId=? and k.start>=? and s.value not like '%nccl%' order by k.start limit 1",(pid,stream,end)).fetchone()
 rows.append({'meta':meta,'cpu_start_ns':ns,'cpu_end_ns':ne,'pid':pid,'gpu_start_ns':start,'gpu_end_ns':end,'stream':stream,'kernel':kernel,'next_same_stream_non_nccl':nextk})
with open(sys.argv[2]+'.jsonl','w') as f:
 for row in rows:f.write(json.dumps(row)+'\n')
groups=collections.defaultdict(list)
for r in rows:
 m=r['meta'];kind=m['op']+'/'+m['phase']; kind+=('/tokens' if len(m['shape'])==2 else '/probs') if m['op']=='token_dispatch' else ''
 groups[kind].append((r['gpu_end_ns']-r['gpu_start_ns'])/1000)
out={k:{'n':len(v),'median_us':statistics.median(v),'mean_us':statistics.mean(v),'min_us':min(v),'max_us':max(v)} for k,v in groups.items()}
out['rows']=len(rows); out['streams']=dict(collections.Counter(str(r['stream']) for r in rows))
print(json.dumps(out,indent=2))
