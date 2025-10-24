# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:39:13 2023

@author: giuse
"""

import numpy as np
from tools import Model
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from os import listdir
from os.path import isfile, join


maxFigWidth=13.7
maxFigHeight=19.3

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def load_chains():
    path=os.getcwd()+'\\chains'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    file_chains=[]
    for file in files:
        if file.startswith('chains_'):
            file_chains.append(file)
            
    file_chains.sort(key= lambda x: float((x.strip('chains_par_')).strip('.txt')))
    
    nDims=len(file_chains)
    file=open(path+'\\'+file_chains[0])
    f=file.read().splitlines()
    nSteps=len(f)
    nChains=len(f[1].split())
    
    chains=np.zeros((nSteps,nChains,nDims))
    
    for i in range(nDims):
        filename=path+'\\'+file_chains[i]
        file=open(filename)
        f=file.read().splitlines()
        for j in range(nSteps):
            chains[j,:,i]=f[j].split()[:nChains]
        file.close()    
    return chains

def elasticity_sample(
    thetas,
    rel_step=0.10,            # float or (d,) array of relative steps
    include_base=True,
    bounds=None,              # <-- 2D array of shape (d,2): [lb, ub]; use np.nan for no bound
    dtype=float,
    max_shrinks=10
    ):
    
    thetas = np.asarray(thetas, dtype=float)
    if thetas.ndim != 2:
        raise ValueError("`thetas` must be (N, d).")
    N, d = thetas.shape

    # rel_step -> (d,)
    rel_arr = np.full(d, rel_step, dtype=float) if np.isscalar(rel_step) else np.asarray(rel_step, dtype=float)
    if rel_arr.shape != (d,):
        raise ValueError("`rel_step` must be a float or a 1D array of shape (d,).")
    if np.any(rel_arr <= 0):
        raise ValueError("All relative steps must be > 0.")

    # --- NEW: normalize 2D bounds array (d,2) with NaN for no bound
    if bounds is None:
        lb = np.full(d, -np.inf, dtype=float)
        ub = np.full(d,  np.inf, dtype=float)
    else:
        b = np.asarray(bounds, dtype=float)
        if b.shape != (d, 2):
            raise ValueError("`bounds` must be a (d, 2) array of [lb, ub].")
        lb = b[:, 0].copy()
        ub = b[:, 1].copy()
        # Replace NaNs with ±inf
        lb = np.where(np.isnan(lb), -np.inf, lb)
        ub = np.where(np.isnan(ub),  np.inf, ub)
        if np.any(lb > ub):
            raise ValueError("Each lower bound must be ≤ upper bound.")

    block = (1 + 2*d) if include_base else (2*d)
    batch = np.empty((N * block, d), dtype=dtype)

    sample_slices = []
    base_idx  = np.full(N, -1, dtype=int)
    plus_idx  = np.empty((N, d), dtype=int)
    minus_idx = np.empty((N, d), dtype=int)
    rel_used  = np.empty((N, d), dtype=float)

    for i in range(N):
        theta = thetas[i]
        start = i * block
        pos = start

        if include_base:
            # clamp baseline just in case
            theta0 = np.minimum(np.maximum(theta, lb), ub)
            batch[pos] = theta0
            base_idx[i] = pos
            pos += 1
        else:
            theta0 = np.minimum(np.maximum(theta, lb), ub)

        for j in range(d):
            rel = float(rel_arr[j])

            # shrink rel so that both +/- stay within [lb[j], ub[j]]
            k = 0
            while k < max_shrinks:
                up = theta0[j] * (1.0 + rel)
                dn = theta0[j] * (1.0 - rel)
                if (up <= ub[j]) and (dn >= lb[j]):
                    break
                rel *= 0.5
                k += 1
            rel_used[i, j] = rel

            # + perturb
            t_plus = theta0.copy()
            t_plus[j] = np.minimum(theta0[j] * (1.0 + rel), ub[j])
            batch[pos] = t_plus
            plus_idx[i, j] = pos
            pos += 1

            # - perturb
            t_minus = theta0.copy()
            t_minus[j] = np.maximum(theta0[j] * (1.0 - rel), lb[j])
            batch[pos] = t_minus
            minus_idx[i, j] = pos
            pos += 1

        sample_slices.append(slice(start, start + block))

    info = dict(
        N=N, d=d, rel_step=rel_arr, block_size=block,
        sample_slices=sample_slices, base_idx=base_idx,
        plus_idx=plus_idx, minus_idx=minus_idx, rel_used=rel_used
    )
    return batch, info


def elasticity_analyze(y_batch,info,positive_output=True,eps=1e-12,
):
    """
    Compute per-parameter mechanistic elasticities from batched model outputs,
    keeping only finite values (excludes non-convergent runs).
    """
    y_batch = np.asarray(y_batch, dtype=float)
    N, d = int(info["N"]), int(info["d"])
    base_idx  = info["base_idx"]
    plus_idx  = info["plus_idx"]   # (N, d)
    minus_idx = info["minus_idx"]  # (N, d)
    rel_used  = info["rel_used"]   # (N, d)

    # Ensure y_batch has shape (batch_size, M)
    if y_batch.ndim == 1:
        y_batch = y_batch[:, None]
    batch_size, M = y_batch.shape

    # Broadcast positive_output to (M,)
    if np.isscalar(positive_output):
        pos_mask = np.full(M, bool(positive_output))
    else:
        pos_mask = np.asarray(positive_output, dtype=bool)
        if pos_mask.shape != (M,):
            raise ValueError("`positive_output` must be a bool or a 1D array of shape (M,).")

    # Baseline outputs y0: (N, M). If no base rows, fill with NaN; we'll fallback to midpoint when needed.
    y0 = np.full((N, M), np.nan)
    if np.all(base_idx >= 0):
        y0 = y_batch[base_idx, :]

    # Output containers
    E_mat = np.full((N, d, M), np.nan, dtype=float)

    # Precompute denominator for log–log central difference per (i,j)
    # den = log(1+rel) - log(1-rel)  -> shape (N, d)
    den_log = np.log1p(rel_used) - np.log1p(-rel_used)

    for j in range(d):
        # y_plus / y_minus for all i: shapes (N, M)
        y_plus  = y_batch[plus_idx[:, j], :]
        y_minus = y_batch[minus_idx[:, j], :]

        # Masks for finite y_plus/y_minus
        fin_pm = np.isfinite(y_plus) & np.isfinite(y_minus)

        # ----- Branch A: log–log elasticity for objectives with positive_output=True
        if np.any(pos_mask):
            # For log–log we require y_plus>0 and y_minus>0
            pos_ok = (y_plus > 0.0) & (y_minus > 0.0)
            mask_A = fin_pm & pos_ok
            if np.any(mask_A):
                num = np.zeros_like(y_plus)
                # Use safe logs only where valid
                num[mask_A] = np.log(np.clip(y_plus[mask_A], eps, None)) - np.log(np.clip(y_minus[mask_A], eps, None))
                # Broadcast den_log[:, j] to (N, M)
                denA = np.repeat(den_log[:, [j]], M, axis=1)
                E_A = np.full_like(y_plus, np.nan, dtype=float)
                # Avoid division by zero in denominator
                good_den = np.abs(denA) > eps
                ok = mask_A & good_den
                E_A[ok] = num[ok] / denA[ok]
                # write back only to objectives with pos_mask==True
                for m in range(M):
                    if pos_mask[m]:
                        E_mat[:, j, m] = np.where(np.isfinite(E_A[:, m]), E_A[:, m], E_mat[:, j, m])

        # ----- Branch B: normalized central difference for objectives with positive_output=False
        if np.any(~pos_mask):
            # y0 to use: baseline if available else midpoint
            if np.all(base_idx < 0):
                y0_use = 0.5 * (y_plus + y_minus)
            else:
                y0_use = y0

            # finiteness mask includes y0
            fin_B = fin_pm & np.isfinite(y0_use)

            # denominator: 2 * rel * y0
            denB = 2.0 * rel_used[:, [j]] * y0_use  # shape (N, M)
            # avoid tiny denom
            good_denB = np.isfinite(denB) & (np.abs(denB) > eps)

            mask_B = fin_B & good_denB
            if np.any(mask_B):
                E_B = np.full_like(y_plus, np.nan, dtype=float)
                diff = y_plus - y_minus
                E_B[mask_B] = diff[mask_B] / denB[mask_B]
                for m in range(M):
                    if not pos_mask[m]:
                        E_mat[:, j, m] = np.where(np.isfinite(E_B[:, m]), E_B[:, m], E_mat[:, j, m])

    # Summaries per parameter and objective
    E_values = [[None for _ in range(M)] for __ in range(d)]
    E_median = np.full((d, M), np.nan, dtype=float)
    E_q25    = np.full((d, M), np.nan, dtype=float)
    E_q75    = np.full((d, M), np.nan, dtype=float)
    finite_frac = np.zeros((d, M), dtype=float)

    for j in range(d):
        for m in range(M):
            v = E_mat[:, j, m]
            v_fin = v[np.isfinite(v)]
            E_values[j][m] = v_fin
            if v_fin.size:
                E_median[j, m] = np.median(v_fin)
                E_q25[j, m]    = np.percentile(v_fin, 25)
                E_q75[j, m]    = np.percentile(v_fin, 75)
                finite_frac[j, m] = v_fin.size / N
            else:
                E_median[j, m] = np.nan
                E_q25[j, m]    = np.nan
                E_q75[j, m]    = np.nan
                finite_frac[j, m] = 0.0

    return {
        "E_median": E_median,             # (d, M)
        "E_IQR": np.stack([E_q25, E_q75], axis=-1),  # (d, M, 2)
        "finite_frac": finite_frac,       # (d, M)
        "E_values": E_values,             # list[d][m] -> 1D array
        "E_matrix": E_mat,                # (N, d, M)
    }



def elasticity_evaluate(model,sample):
    N=len(sample)
    l=np.zeros((N,4))
    for i in range(N):
        r=[]
        func(sample[i,:],model,r,0)
        l[i,:]=r[0][1]
    return l

def func(params,mod,results,idx):
    mod.overwrite_params(params)
    mod.execute()
    df=pd.read_csv(mod.path_project+r'\SUGAR.out',sep='\s+',skiprows=2,header=None,index_col=0)
    if df.index.values[-1]!=2256.:
        obj_list=[np.nan,np.nan,np.nan,np.nan]
    else:        
        obj_list=df.iloc[-1,[1,2,3,4]].values
    results.append([idx,obj_list,0])
    

def plot_elasticity_dist(res, qclip=(5, 95)):
    """
    Stack violin plots (one per objective) vertically.
    
    Parameters
    ----------
        Percentile range to set y-limits per objective (e.g., (5,95)).
    """
    names=[r'$\theta_{r}$', r'$\alpha$', r'$n$',
            r'$K_{s}$', r'$k_{rs}^{max}$', r'$k_{comp}^{max}$',
            r'$k$', r'$a$',r'$v_{m}$', r'$\delta$', r'$\mu_{st}$', 
            r'$\beta_{st}$', r'$mu_{syn,so}$']
    d = len(names)
    # infer M from structure
    M = len(res["E_values"][0])

    titles=['Water mass at harvesting', 'Dry mass at harvesting', 
            'Soluble sugar mass at harvesting', 'Starch mass at harvesting']

    figsize=cm2inch(maxFigWidth,maxFigWidth)

    fig, axes = plt.subplots(M, 1, sharex=True, figsize=figsize, dpi=200)
    if M == 1:
        axes = [axes]

    positions = np.arange(1, d + 1)

    for m, ax in enumerate(axes):
        data = []
        empty_idx = []
        for j in range(d):
            v = np.asarray(res["E_values"][j][m])
            v = v[np.isfinite(v)]
            if v.size == 0:
                empty_idx.append(j)
                data.append(np.array([np.nan]))
            else:
                data.append(v)
        pooled = np.concatenate([x[np.isfinite(x)] for x in data]) if any(np.isfinite(x).any() for x in data) else np.array([])
        if pooled.size:
            lo, hi = np.percentile(pooled, qclip)
            pad = 0.1 * (hi - lo + 1e-12)
            ylims = (lo - pad, hi + pad)
        else:
            ylims = (-1, 1)

        vp = ax.violinplot(
            data,
            positions=positions,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )
        
        ax.set_ylim(*ylims)
        ax.grid(axis='y', linestyle=':', linewidth=0.8)
        ax.set_ylabel("Elasticity (-)")
        ax.set_title(titles[m])

    # x-axis labels only once (bottom subplot), keep original order
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(names, rotation=60, ha='right')
    fig.tight_layout()
    return fig, axes
  
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
  
    
treatment='LI'
model_name=treatment
HYDRUS_path=os.getcwd()+'\\HYDRUS_models\\'+model_name


#------------------CALIBRATED FRUIT PARAMS-------------------------------

Krs = {'name':'Krs','x0':1e-4, 'bounds': np.array([5e-8,5e-4]), 'flags': np.array([1])} 
Kcomp = {'name':'Kcomp', 'x0':1e-7, 'bounds': np.array([5e-8,5e-4]), 'flags': np.array([1])}
hxMin = {'name':'hxMin','x0':-16000, 'bounds': np.array([-20000,-8000]), 'flags': np.array([0])} 
gamma = {'name':'gamma','x0':5.57, 'bounds': np.array([4,6]), 'flags': np.array([0])} 
eta = {'name':'eta','x0':0.62, 'bounds': np.array([0.3,0.8]), 'flags': np.array([0])} 
qg = {'name':'qg','x0':0.22, 'bounds': np.array([0.1,0.5]), 'flags': np.array([0])} 
qm = {'name':'qm','x0':0.00042, 'bounds': np.array([0.0001,0.01]), 'flags': np.array([0])} 
Q10 = {'name':'Q10','x0':1.47, 'bounds': np.array([1,2.5]), 'flags': np.array([0])} 
ps = {'name':'ps','x0':3.6e-5, 'bounds': np.array([1e-5,1e-2]), 'flags': np.array([0])} 
Cp = {'name':'Cp','x0':0.11, 'bounds': np.array([0.08,0.15]), 'flags': np.array([0])} 
phi_max = {'name':'phi_max', 'x0':0.02, 'bounds': np.array([0.001,0.1]), 'flags': np.array([0])} 
k = {'name':'k','x0':0.0044, 'bounds': np.array([1e-5,0.05]), 'flags': np.array([1])} 
tau = {'name':'tau', 'x0':3.38e-6, 'bounds': np.array([1e-8,1e-4]), 'flags': np.array([0])} 
Y = {'name':'Y','x0':1, 'bounds': np.array([0.1,10.]), 'flags': np.array([0])} 
a = {'name':'a', 'x0':0.0116, 'bounds': np.array([0.001,0.05]), 'flags': np.array([1])} 
Lp = {'name':'Lp','x0':0.015, 'bounds': np.array([0.001,0.05]), 'flags': np.array([0])} 
Lx = {'name':'Lx', 'x0':Lp['x0'], 'bounds': np.array([0.001,0.05]), 'flags': np.array([0])} 
vm = {'name':'vm','x0':2.64, 'bounds': np.array([0.1,5]), 'flags': np.array([1])} 
Km = {'name':'Km', 'x0':0.08, 'bounds': np.array([0.01,1]), 'flags': np.array([0])} 
fstar = {'name':'fstar', 'x0': 0.45, 'bounds': np.array([0,1]), 'flags': np.array([0])} 
delta = {'name':'delta','x0':579, 'bounds': np.array([1,2000]), 'flags': np.array([1])} 
ro = {'name':'ro', 'x0': 23.4, 'bounds': np.array([1,1000]), 'flags': np.array([0])} 
kst = {'name':'kst','x0':0.14, 'bounds': np.array([0.001,2]), 'flags': np.array([1])} 
kst50 = {'name':'kst50', 'x0':760, 'bounds': np.array([1,2000]), 'flags': np.array([0])} 
beta_st = {'name':'beta_st','x0':4.95, 'bounds': np.array([1,5]), 'flags': np.array([1])} 
kdeg_st = {'name':'kdeg_st', 'x0':0.0187, 'bounds': np.array([0.001,2]), 'flags': np.array([0])} 
ksyn_so = {'name':'ksyn_so','x0':5e-4, 'bounds': np.array([0.0,0.1]), 'flags': np.array([1])}

par_list=[Krs,Kcomp,hxMin,gamma,eta,qg,qm,Q10,ps,Cp,phi_max,k,tau,Y,a,Lp,Lx,vm,Km,delta,fstar,ro,kst,kst50,beta_st,kdeg_st,ksyn_so]

model=Model(HYDRUS_path,additional_params_list=par_list,
            execution_time=20)

chains=load_chains()
nSteps,nChains,nDims=chains.shape
L=int(0.1*nSteps)
cshort=chains[-L:,:,:]
flat=np.reshape(cshort,(L*nChains,nDims))
mean=np.mean(flat,axis=0)
D=200

idx=idx=np.random.choice(np.arange(L*nChains),D,replace=False)
thetas=flat[idx]


lb=model.convert_dict_to_pars(model.lb)
ub=model.convert_dict_to_pars(model.ub)
bounds=np.vstack((lb,ub)).T
sample, info = elasticity_sample(
    thetas, rel_step=0.05, include_base=True, bounds=bounds
)

y = elasticity_evaluate(model, sample)
res = elasticity_analyze(y, info, positive_output=True)
plot_elasticity_dist(res)






