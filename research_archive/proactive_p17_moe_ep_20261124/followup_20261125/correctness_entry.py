"""One-step full-tensor diagnostic. Explicitly invoked; not a performance run."""
import os, runpy, random
from pathlib import Path
import numpy as np
import torch
import megatron.training.training as training
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import _get_all_rng_states
from megatron.core.transformer.moe.router import TopKRouter

out=Path(os.environ['P17_VERIFY_OUT']); rank=int(os.environ['RANK']); out.mkdir(parents=True,exist_ok=True)
old_step=training.train_step

def cpu(t): return t.detach().cpu().clone() if torch.is_tensor(t) else t

def step(*args, **kwargs):
    if kwargs.get('iteration') != 0: return old_step(*args, **kwargs)
    models, opt=args[2],args[3]
    params={f'{i}:{n}':p for i,m in enumerate(models) for n,p in m.named_parameters() if p.requires_grad}
    initial={n:cpu(p) for n,p in params.items()}
    torch.save({'parameters':initial,'optimizer':opt.state_dict(),'rng':_get_all_rng_states(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},out/f'initial.rank{rank}.pt')
    observed={'inputs':[], 'outputs':[], 'routes':[]}
    handles=[]
    def pre(m,a,k):
        observed['inputs'].append({'tokens':cpu(a[0]),'position_ids':cpu(a[1]),'attention_mask':cpu(a[2]),'labels':cpu(k.get('labels')),'loss_mask':cpu(k.get('loss_mask'))})
    def post(m,a,k,o): observed['outputs'].append(cpu(o))
    def route(m,a,o): observed['routes'].append({'layer':m.layer_number,'mapping':cpu(o[1])})
    for model in models:
        for m in model.modules():
            if isinstance(m,GPTModel):
                handles.extend([m.register_forward_pre_hook(pre,with_kwargs=True),m.register_forward_hook(post,with_kwargs=True)])
            if isinstance(m,TopKRouter): handles.append(m.register_forward_hook(route))
    oldopt=opt.step
    def optstep(*a,**k):
        observed['gradients']={n:cpu(p.main_grad) for n,p in params.items() if getattr(p,'main_grad',None) is not None}
        observed['missing_gradients']=sorted(set(params)-set(observed['gradients']))
        ret=oldopt(*a,**k)
        observed['parameters']={n:cpu(p) for n,p in params.items()}
        return ret
    opt.step=optstep
    try:
        result=old_step(*args,**kwargs)
        observed['loss']={str(k):cpu(v) for k,v in result[0].items()}
        torch.save(observed,out/f'observed.rank{rank}.pt')
        return result
    finally:
        opt.step=oldopt
        for h in handles:h.remove()
training.train_step=step
runpy.run_path('pretrain_gpt.py',run_name='__main__')
