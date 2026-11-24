import sqlite3,json,collections
from pathlib import Path
r=Path(__file__).resolve().parent;p=r/'p14_4b_C0_trace_conn1_pp2';db=sqlite3.connect(p/'trace.sqlite'); ranks=collections.defaultdict(list)
for a,b,t in db.execute("select start,end,text from NVTX_EVENTS where text like 'P10|%' and end is not null"):
 x=json.loads(t[4:]);x.update(start=a,end=b);ranks[x['rank']].append(x)
out={}
for rank,rows in ranks.items():
 forward=sorted((x for x in rows if x['kind']=='forward'),key=lambda x:x['start']); backward=sorted((x for x in rows if x['kind']=='backward'),key=lambda x:x['start']); p2p=sorted((x for x in rows if x['kind']=='p2p'),key=lambda x:x['start']);qkv=[x for x in rows if x['kind']=='qkv']
 assert len(forward)==len(backward)==16
 assert [x['stage'][1] for x in forward]==list(range(16))
 assert [x['stage'][1] for x in backward]==list(range(16))
 for e in qkv:
  if e['stage'][0]=='forward':assert any(x['start']<=e['start']<e['end']<=x['end'] and x['stage']==e['stage'] for x in forward)
 out[rank]=dict(pp_rank=p2p[0]['pp_rank'],forward_count=len(forward),backward_count=len(backward),forward_layers=sorted(set(x['layer'] for x in qkv if x['stage'][0]=='forward')),p2p_counts=dict(collections.Counter(x['method'] for x in p2p)),events=p2p,identities='stage and counters recorded; P2P payload-to-microbatch and Backward graph identity not proven')
(r/'pp2_audit.json').write_text(json.dumps(out,indent=2));print({k:{x:y for x,y in v.items() if x!='events'} for k,v in out.items()})
