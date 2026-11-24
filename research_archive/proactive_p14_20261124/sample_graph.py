import json
from pathlib import Path
root=Path(__file__).resolve().parent
p=root.parent/'proactive_p13_20261124'/'p13_4b_C0_trace_conn1_tp2_event'
marks=json.loads((p/'nvtx_attribution.json').read_text()); audit=json.loads((root/(p.name+'_event_audit.json')).read_text())
# Retain only exact producer association, then require membership of a TP2 API scope.
lookup={(x['wait']['pid'],x['kernel']['corr']):x for x in audit['matched_nccl']}
out=[]
for rank in (0,1):
 rr=marks[str(rank)]; scopes=sorted((x for x in rr if x['kind']=='all_reduce_api' and x['stage']==['backward',7] and x.get('peers')==[0,1]),key=lambda x:x['cpu_start'])
 choices=[]
 for ix,e in enumerate(scopes):
  ks=[k for k in e['kernels'] if 'allreduce' in k['name'].lower().replace('_','') and (k['pid'],k['correlation']) in lookup]
  if len(ks)==1:
   k=ks[0];x=lookup[k['pid'],k['correlation']]; w=x['wait']
   if e['cpu_start'] <= w['record']['start'] <= w['wait_start'] < e['cpu_end']:
    choices.append((ix,e,k,x))
 ix,e,k,x=choices[0]; orig=[q for q in rr if q['kind']=='qkv' and q['stage']==['forward',7] and q['layer']==1]; back=[q for q in rr if q['kind']=='qkv' and q['stage']==['backward',7] and q['layer']==1]
 assert len(orig)==len(back)==1
 q=orig[0]; b=back[0]
 out.append(dict(rank=rank,tp_peers=[0,1],collective_ordinal=ix,api_cpu_start=e['cpu_start'],api_cpu_end=e['cpu_end'],kernel=k,event=x, candidate=dict(layer=1,mb=7,original_qkv_gpu_start=min(v['start'] for v in q['kernels']), original_qkv_gpu_end=max(v['end'] for v in q['kernels']),replay_gpu_start=min(v['start'] for v in b['kernels']), replay_gpu_end=max(v['end'] for v in b['kernels'])),unknown=['collective input producer','first result consumer','direct dependence of candidate input on collective','cross-rank event identity alignment']))
(root/'sample_graph.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
