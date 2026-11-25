import sqlite3,json,sys,collections,bisect,statistics
c=sqlite3.connect(sys.argv[1]); by=collections.defaultdict(list)
for pid,a,b,stream,name in c.execute("select k.globalPid,k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where lower(s.value) not like '%nccl%'"): by[pid].append((a,b,stream,name))
merged={}
for pid,records in by.items():
 out=[]
 for a,b,stream,name in sorted(records):
  if out and a<=out[-1][1]: out[-1]=(out[-1][0],max(b,out[-1][1]))
  else:out.append((a,b))
 merged[pid]=(out,[b for a,b in out],[a for a,b in out])
group=collections.defaultdict(list)
for line in open(sys.argv[2]):
 r=json.loads(line);m=r['meta'];a,b=r['gpu_start_ns'],r['gpu_end_ns'];v,ends,starts=merged[r['pid']];i=bisect.bisect_right(ends,a); j=i;overlap=0
 while j<len(v) and v[j][0]<b:
  overlap+=max(0,min(b,v[j][1])-max(a,v[j][0]));j+=1
 before=v[i-1][1] if i else None; nextidx=bisect.bisect_left(starts,b); after=v[nextidx][0] if nextidx<len(v) else None
 kind=m['op']+'/'+m['phase'];kind+=('/tokens' if len(m['shape'])==2 else '/probs') if m['op']=='token_dispatch' else ''
 group[kind].append({'residence':b-a,'overlap':overlap,'gap_before_us':(a-before)/1000 if before is not None else None,'gap_after_us':(after-b)/1000 if after is not None else None})
out={}
for k,vals in group.items():
 res=sum(x['residence'] for x in vals);ov=sum(x['overlap'] for x in vals)
 out[k]={'count':len(vals),'non_nccl_overlap_fraction':ov/res if res else None,'no_overlap_count':sum(x['overlap']==0 for x in vals),'median_non_nccl_gap_before_us':statistics.median(x['gap_before_us'] for x in vals if x['gap_before_us'] is not None),'median_non_nccl_gap_after_us':statistics.median(x['gap_after_us'] for x in vals if x['gap_after_us'] is not None)}
print(json.dumps(out,indent=2))
