"""All tensor elements; fixed tolerance predetermined from previous P17 report."""
import sys,json,math,torch,numpy as np
from pathlib import Path
torch.set_num_threads(4)
a,b=map(Path,sys.argv[1:3]); result=[]

def flatten(obj,p=''):
 if isinstance(obj,torch.Tensor): yield p,obj
 elif isinstance(obj,np.ndarray): yield p,torch.from_numpy(obj.copy())
 elif isinstance(obj,dict):
  for k in sorted(obj,key=str): yield from flatten(obj[k],p+'/'+str(k))
 elif isinstance(obj,(list,tuple)):
  for i,x in enumerate(obj): yield from flatten(x,p+'/'+str(i))
 elif isinstance(obj,(str,int,float,bool,type(None))): yield p,obj
 else: yield p,repr(obj)

def compare(x,y):
 xx=dict(flatten(x)); yy=dict(flatten(y)); missing=sorted(set(xx)^set(yy)); sections={}; details=[]
 for key in sorted(set(xx)&set(yy)):
  x,y=xx[key],yy[key]; category=key.split('/')[1] if '/' in key else key
  row=sections.setdefault(category,{'tensors':0,'elements':0,'different':0,'violations':0,'nonfinite':0,'max_abs':0.,'sumsq_diff':0.,'sumsq_ref':0.,'abs_diff_bins':{'zero':0,'le_1e-7':0,'le_1e-6':0,'le_1e-5':0,'le_1e-4':0,'gt_1e-4':0},'scalar_mismatches':0})
  if not torch.is_tensor(x) or not torch.is_tensor(y):
   if x!=y: row['scalar_mismatches']+=1; details.append({'name':key,'error':'scalar mismatch'})
   continue
  if x.shape!=y.shape or x.dtype!=y.dtype:
   details.append({'name':key,'error':f'shape/dtype mismatch {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}'});continue
  row['tensors']+=1; row['elements']+=x.numel()
  if torch.equal(x.contiguous().reshape(-1).view(torch.uint8),y.contiguous().reshape(-1).view(torch.uint8)): row['abs_diff_bins']['zero']+=x.numel(); continue
  xf=x.float(); yf=y.float(); diff=(xf-yf).abs(); neq=int(torch.count_nonzero(x!=y)); violations=int(torch.count_nonzero(diff > (1e-5+1e-2*xf.abs())))
  nonfinite=int((~torch.isfinite(xf)).sum()+(~torch.isfinite(yf)).sum()); maxabs=float(diff.max()) if diff.numel() else 0.
  row['different']+=neq;row['violations']+=violations;row['nonfinite']+=nonfinite;row['max_abs']=max(row['max_abs'],maxabs)
  row['sumsq_diff']+=float(torch.sum(diff.double().square())); row['sumsq_ref']+=float(torch.sum(xf.double().square()))
  bins=row['abs_diff_bins']; bins['zero']+=int((diff==0).sum()); bins['le_1e-7']+=int(((diff>0)&(diff<=1e-7)).sum());bins['le_1e-6']+=int(((diff>1e-7)&(diff<=1e-6)).sum());bins['le_1e-5']+=int(((diff>1e-6)&(diff<=1e-5)).sum());bins['le_1e-4']+=int(((diff>1e-5)&(diff<=1e-4)).sum());bins['gt_1e-4']+=int((diff>1e-4).sum())
  details.append({'name':key,'elements':x.numel(),'different':neq,'violations':violations,'max_abs':maxabs})
 for v in sections.values():
  v['relative_l2_differing_tensors']=math.sqrt(v.pop('sumsq_diff')/v.pop('sumsq_ref')) if v['sumsq_ref'] else None
 return {'missing_keys':missing,'sections':sections,'differing_tensors':details}
for rank in range(8):
 row={'rank':rank}
 for name in ('initial','observed'):
  paths=[d/f'{name}.rank{rank}.pt' for d in (a,b)]
  if not all(p.exists() for p in paths): row[name]={'error':'missing snapshot'};continue
  row[name]=compare(*(torch.load(p,map_location='cpu',weights_only=False) for p in paths))
 result.append(row)
print(json.dumps(result,indent=2,allow_nan=False))
