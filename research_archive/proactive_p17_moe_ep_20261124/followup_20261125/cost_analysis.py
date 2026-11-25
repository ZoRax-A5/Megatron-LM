"""Steady 0-based steps 20..22, per-process CPU ranges and GPU kernel sums."""
import sqlite3,json,sys,collections,statistics,os
c=sqlite3.connect(sys.argv[1]); steps=[]
steps_chosen=set(map(int,os.getenv('P17_STEPS','20,21,22').split(',')))
for txt,start,end,tid in c.execute("select text,start,end,globalTid from NVTX_EVENTS where text like 'P17STEP_%'"):
 if int(txt.split('_')[1]) not in steps_chosen:continue
 steps.append({'step':int(txt.split('_')[1]),'start':start,'end':end,'pid':tid & ~16777215,'step_cpu_ms':(end-start)/1e6,'gpu_gemm_ms':0.,'gpu_nccl_ms':0.,'gpu_other_ms':0.,'gpu_kernel_count':0,'cuda_runtime_ms':0.,'cuda_synchronize_ms':0.,'checkpoint_backward_cpu_ms':0.,'checkpoint_replay_cpu_ms':0.,'checkpoint_replay_calls':0})
by=collections.defaultdict(list)
for s in steps:by[s['pid']].append(s)
def locate(pid,t):
 for s in by.get(pid,[]):
  if s['start']<=t<s['end']:return s
for pid,a,b,name in c.execute('select k.globalPid,k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName'):
 s=locate(pid,a)
 if s is None:continue
 n=name.lower(); key='gpu_nccl_ms' if 'nccl' in n else 'gpu_gemm_ms' if 'gemm' in n else 'gpu_other_ms';s[key]+=(b-a)/1e6;s['gpu_kernel_count']+=1
for tid,a,b,name in c.execute('select r.globalTid,r.start,r.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId'):
 s=locate(tid & ~16777215,a)
 if s is None:continue
 s['cuda_runtime_ms']+=(b-a)/1e6
 if 'Synchronize' in name:s['cuda_synchronize_ms']+=(b-a)/1e6
for tid,a,b,name in c.execute("select globalTid,start,end,text from NVTX_EVENTS where text in ('P17_CHECKPOINT_BACKWARD','P17_checkpoint_replay')"):
 s=locate(tid & ~16777215,a)
 if s is None:continue
 if name=='P17_CHECKPOINT_BACKWARD':s['checkpoint_backward_cpu_ms']+=(b-a)/1e6
 else:s['checkpoint_replay_cpu_ms']+=(b-a)/1e6;s['checkpoint_replay_calls']+=1
print(json.dumps({'steps':steps,'means':{key:statistics.mean(s[key] for s in steps) for key in ('step_cpu_ms','gpu_gemm_ms','gpu_nccl_ms','gpu_other_ms','gpu_kernel_count','cuda_runtime_ms','cuda_synchronize_ms','checkpoint_backward_cpu_ms','checkpoint_replay_cpu_ms','checkpoint_replay_calls')}},indent=2))
