import numpy as np

def jitter(x, sigma=0.02): return x + np.random.normal(0.0, sigma, x.shape)
def scaling(x, sigma=0.1):
    factor = np.random.normal(1.0, sigma, size=(x.shape[0],1,x.shape[2]))
    return x*factor
def permute(x, n_perm=4):
    b,t,f=x.shape; seg=t//n_perm; out=[]
    for i in range(b):
        idx=np.arange(n_perm); np.random.shuffle(idx)
        parts=[x[i,j*seg:(j+1)*seg,:] for j in idx]
        out.append(np.concatenate(parts, axis=0))
    return np.stack(out, axis=0)
def time_warp(x, sigma=0.2):
    b,t,f=x.shape; out=np.zeros_like(x)
    for i in range(b):
        sp=np.clip(np.random.normal(1.0,sigma),0.7,1.3)
        idx=(np.arange(t)*sp).astype(int); idx[idx>=t]=t-1
        out[i]=x[i,idx,:]
    return out
def apply_augs(x, use_jitter=True, use_scaling=True, use_permute=False, use_timewarp=False):
    if use_jitter: x=jitter(x)
    if use_scaling: x=scaling(x)
    if use_permute: x=permute(x)
    if use_timewarp: x=time_warp(x)
    return x
