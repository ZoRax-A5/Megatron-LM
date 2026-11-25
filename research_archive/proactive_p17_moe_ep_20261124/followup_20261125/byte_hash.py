"""Hash complete raw tensor bytes and metadata by category and rank."""
import sys,json,hashlib,torch
from pathlib import Path
import numpy as np
def flatten(obj,p=''):
 if isinstance(obj,torch.Tensor): yield p,obj
 elif isinstance(obj,np.ndarray): yield p,torch.from_numpy(obj.copy())
 elif isinstance(obj,dict):
  for k in sorted(obj,key=str): yield from flatten(obj[k],p+'/'+str(k))
 elif isinstance(obj,(list,tuple)):
  for i,x in enumerate(obj): yield from flatten(x,p+'/'+str(i))
root=Path(sys.argv[1]); result={}
for mode in ('det1','det2','det_fc1'):
 result[mode]=[]
 for rank in range(8):
  by={}
  for stage in ('initial','observed'):
   obj=torch.load(root/mode/f'{stage}.rank{rank}.pt',map_location='cpu',weights_only=False)
   for name,t in sorted(flatten(obj),key=lambda x:x[0]):
    if not torch.is_tensor(t):continue
    key=stage+'/'+name.split('/')[1]; h=by.setdefault(key,hashlib.sha256())
    h.update(json.dumps([name,list(t.shape),str(t.dtype)]).encode());h.update(memoryview(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy()))
  result[mode].append({k:v.hexdigest() for k,v in by.items()})
print(json.dumps(result,indent=2))
