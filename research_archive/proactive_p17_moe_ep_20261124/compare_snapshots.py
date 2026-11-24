import json, sys, torch
from pathlib import Path
root=Path(sys.argv[1]); result=[]; second=sys.argv[2] if len(sys.argv)>2 else 'fc1'
for rank in range(8):
    files=[root/f'ep2_snapshot_{mode}.rank{rank}.pt' for mode in ('baseline',second)]
    if not all(p.exists() for p in files):
        result.append({'rank':rank,'error':'missing snapshot'}); continue
    a,b=(torch.load(p,map_location='cpu',weights_only=True) for p in files)
    row={'rank':rank,'missing_gradients':[a['missing_gradients'],b['missing_gradients']]}
    for kind in ('gradients','parameters'):
        x,y=a[kind],b[kind]
        if x.keys()!=y.keys(): row[kind]={'error':'name mismatch'}; continue
        elements=0; max_abs=0.; sum_sq=0.; ref_sq=0.; nonfinite=0; allclose=True
        for key in sorted(x):
            t=x[key].float(); u=y[key].float()
            if t.shape!=u.shape: raise ValueError(key)
            d=(t-u).double(); elements+=t.numel(); max_abs=max(max_abs,d.abs().max().item() if d.numel() else 0)
            sum_sq+=(d*d).sum().item(); ref_sq+=(t.double()**2).sum().item()
            nonfinite+=int((~torch.isfinite(t)).sum().item()+(~torch.isfinite(u)).sum().item())
            allclose &= torch.allclose(t,u,atol=1e-5,rtol=1e-2)
        row[kind]={'tensors':len(x),'elements':elements,'max_abs':max_abs,'relative_l2':(sum_sq/ref_sq)**0.5 if ref_sq else None,'nonfinite':nonfinite,'allclose_atol_1e-5_rtol_1e-2':allclose}
    result.append(row)
print(json.dumps(result,indent=2))
