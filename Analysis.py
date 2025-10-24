# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:39:13 2023

@author: giuse
"""

import numpy as np
from tools import Model
from mcmc import AIES
    
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

add_par={'sigma1':{'Meas_ID':2,'x0':0.03,'bounds': np.array([0.029,0.031]),'flag':0},
              'sigmaFW':{'Meas_ID':40,'x0':27.95,'bounds': np.array([27,28]),'flag':0},
              'sigmaDW':{'Meas_ID':41,'x0':3.9,'bounds': np.array([3.8,4]),'flag':0},
              'sigmaSS':{'Meas_ID':42,'x0':2.44,'bounds': np.array([2.4,2.5]),'flag':0},
              'sigmaST':{'Meas_ID':43,'x0':0.44,'bounds': np.array([0.4,0.45]),'flag':0}}

sampler=AIES(model,additional_params=add_par)

xtry=np.array([8.01399639e-02, 8.25203480e-02, 1.95148590e+00, 6.08384035e+01,
            5.21207367e-06, 3.31783647e-04, 8.56236208e-03, 3.14916833e-02,
            1.33885680e+00, 8.29076545e+01, 1.93931091e-02, 3.58232002e+00,
            3.5e-4])
    
nChains=2*sampler.nDims
start=sampler.spherical_walkers_init(xtry,xtry*0.01,nChains)
idx=[]
for i in range(nChains):
    for j in range(sampler.nDims):
        while start[i,j]>sampler.ub[j]:
            start[i,j]=np.random.normal(xtry[j],abs(xtry[j])*0.01)
        while start[i,j]<sampler.lb[j]:
            start[i,j]=np.random.normal(xtry[j],abs(xtry[j])*0.01)
            pass
        pass
    pass

nSteps=30000
a=2
Output_folder_name=treatment+'_mcmc'
sampler.run_mcmc(nChains, a, nSteps, start, Output_folder_name)