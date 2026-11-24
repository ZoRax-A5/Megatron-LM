import json,glob,sys,statistics
from collections import defaultdict
out={}
for ep in (2,8):
 layers=defaultdict(list)
 for p in glob.glob(f'{sys.argv[1]}/ep{ep}_routing.rank*.jsonl'):
  for line in open(p):
   d=json.loads(line)
   if 'counts' in d: layers[d['layer']].append(d)
 out[ep]={}
 for layer,records in layers.items():
  totals=[sum(r['counts'][i] for r in records) for i in range(8)]
  ratios=[max(r['counts'])/(sum(r['counts'])/8) for r in records if sum(r['counts'])]
  out[ep][layer]={'observations':len(records),'per_expert_assignments':totals,'max_to_mean_aggregated':max(totals)/(sum(totals)/8),'max_to_mean_observed_range':[min(ratios),max(ratios)],'token_count_sum':sum(r['tokens'] for r in records),'assignments_sum':sum(totals)}
print(json.dumps(out,indent=2))
