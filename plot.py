# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:03:21 2024

@author: WKS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tools import Model
import os
from os import listdir
from os.path import isfile, join
import emcee
import matplotlib.patches as patches


tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "text.latex.preamble" : r'\usepackage{amsmath}',
    "text.latex.preamble" : r'\usepackage[symbol]{footmisc}',    
}

plt.rcParams.update(tex_fonts)

maxFigWidth=13.7
maxFigHeight=19.3

def conv_to_log_cm_bar_h(k):
    tconv=3600 #to seconds
    lconv=1/100 #to meters
    sp=9807 #N/m3 or kg m /s2    
    krs=k/tconv #to 1/s
    krs=krs/sp #to s*m2/kg
    krs=krs/(1/1000000) #to m/MPa*s
            
    krs_bar=krs*(1/lconv)/(10*1/tconv) #to cm/bar*h
    return np.log10(krs_bar)

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
def RMSE(meas,mod):
    resid=(meas['FOS']-mod['FOS']).dropna()
    return (np.sum(resid**2)/resid.size)**0.5


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

def load_measurements(treatment):        
    df_meas=pd.read_excel('measurements.xlsx',sheet_name=treatment+'_mean',index_col=0,header=0)
    df_std=pd.read_excel('measurements.xlsx',sheet_name=treatment+'_std',index_col=0,header=0)       
    return df_meas, df_std
    
def plot_fit(treatment,sample,mean):
    df_meas,df_std=load_measurements(treatment)
    model_name=treatment
    HYDRUS_path=os.getcwd()+'\\HYDRUS_models\\'+model_name

    #------------------CALIBRATED FRUIT PARAMS-------------------------------

    Krs = {'name':'Krs','x0':1e-4, 'bounds': np.array([5e-6,5e-4]), 'flags': np.array([1])} 
    Kcomp = {'name':'Kcomp', 'x0':1e-7, 'bounds': np.array([5e-8,5e-4]), 'flags': np.array([1])}
    hxMin = {'name':'hxMin','x0':-16000, 'bounds': np.array([-20000,-8000]), 'flags': np.array([0])} 
    if treatment=='LI':
        gamma = {'name':'gamma','x0':5.57, 'bounds': np.array([4,6]), 'flags': np.array([0])} 
        eta = {'name':'eta','x0':0.62, 'bounds': np.array([0.3,0.8]), 'flags': np.array([0])} 
    elif treatment=='HI':
        gamma = {'name':'gamma','x0':5.73, 'bounds': np.array([4,6]), 'flags': np.array([0])} 
        eta = {'name':'eta','x0':0.65, 'bounds': np.array([0.3,0.8]), 'flags': np.array([0])}         
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
    if treatment=='HI':
        model.th_tol=0.005
    else:
        model.th_tol=0.001

    fg=plt.figure(figsize=cm2inch(maxFigWidth,maxFigWidth),dpi=200)
    gs=gridspec.GridSpec(5,1)
    
    measW=model.observations[['HO(N)','FOS']][model.observations['ITYPE(N)']==2]
        
    ax=fg.add_subplot(gs[0,:])
    if treatment=='HI':
        label_title='Validation. Irrigation Scenario: HI'
    else:
        label_title='Calibration. Irrigation Scenario: LI'
        
    ax.set_title(label_title,fontweight='bold',fontsize='x-large')
    stdW=0.03
    yplus=measW['FOS']+stdW
    yminus=measW['FOS']-stdW
    ax.fill_between(measW['HO(N)'],yminus,yplus,color='0.5',alpha=0.3, label='Measured')
    # ax.plot(measW['HO(N)'],measW['FOS'],'--',color='k',alpha=0.5,markersize=2)
    ax.set_ylim((0.0,0.35))
    ax.set_xlim((0,2256))
    ax.set_xticks([])
    ax.set_ylabel(r'$\theta$ (-)',fontsize=12)
    
    start = 0
    end=720
    width=end-start
    height=0.35
    rect = patches.Rectangle((start, 0.0), width, height, linewidth=0, edgecolor='k', facecolor='none',alpha=0.2,hatch='///')
    ax.add_patch(rect)
    ax.text(200,0.28,'SPIN-UP')
    ax.text(2120,0.03, r'$\textbf{(A)}$')
    
    label_plot=[r'$\textbf{(B)}$',r'$\textbf{(C)}$',r'$\textbf{(D)}$',r'$\textbf{(E)}$']
    if treatment=='full':
        label_height=[200,15,12,5]
    else:
        label_height=[150,15,7,5]
    names=['WW (g)','DW (g)','SSPF (g)','StaPF (g)']
    names_list=[r'$w$ (g)', r'$s$ (g)', r'$s_{s}$ (g)', r'$s_{t}$ (g)']
    idMeas=40
    for i in range(1,5):
        ax=fg.add_subplot(gs[i,:])
        ax.errorbar(df_meas.loc[:,'Time (hours)'],df_meas.loc[:,names[i-1]],yerr=df_std.loc[:,names[i-1]].values,fmt='o',markersize=4,capsize=5,color='k',label='Measured')         
        ax.set_xlim((0,2256))
        ymax=df_meas.loc[:,names[i-1]].max()
        ax.set_ylim((0,ymax+0.2*ymax))
        ax.set_ylabel(names_list[i-1],fontsize=12)
        rect = patches.Rectangle((start, 0), width, ymax+0.2*ymax, linewidth=0, edgecolor='k', facecolor='none',alpha=0.2,hatch='///')
        ax.add_patch(rect)
        ax.axvline(1044,ls='--',linewidth=1,color='0.5')
        if i!=4:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time (hours)', fontsize=12)
        ax.text(2120,label_height[i-1], label_plot[i-1])
        idMeas+=1
    pass
    
    sample=np.vstack((sample,mean))
    for i in range(len(sample)):
        params=sample[i]
    
        thr,alfa,n,Ks,Krs,Kcomp,k,a,vm,delta,kst,beta_st,ksyn_so=params
        model.overwrite_params(params)
        model.execute()
    
        df_sugar=pd.read_csv(model.path_project+r'\SUGAR.OUT',sep='\s+',skiprows=1,header=0,index_col=0)
        df_hyd=pd.read_csv(model.path_project+r'\Obs_Node.out',sep='\s+',skiprows=10,header=0,usecols=[0,2],index_col=0,skipfooter=1,engine='python')
        
        if i==len(sample)-2:
            label='Modeled'
            color='b'
            alpha=0.3
        elif i==len(sample)-1:
            label='Mean'
            color='r'
            alpha=1
            mod=model.read_model_output()
            modW=mod[['HO(N)','FOS']][mod['ITYPE(N)']==2]
            rmse=RMSE(measW,modW)
        else:
            label='_nolegend_'
            color='b'
            alpha=0.3
        
        ax=fg.axes[0]
        ax.plot(df_hyd.index,df_hyd.iloc[:,0],color=color,alpha=alpha,label=label)
        if i==len(sample)-1:
            ax.text(1400,0.28,r'$RMSE_{mean}$='+str(round(rmse,3)))
        
        idMeas=40
            
        for k in range(1,5):
            ax=fg.axes[k]
            ax.plot(df_sugar.index,df_sugar.iloc[:,k],color=color,alpha=alpha,label=label) 
            if i==len(sample)-1:
                measF=model.observations[['HO(N)','FOS']][model.observations['ITYPE(N)']==idMeas]
                modF=mod[['HO(N)','FOS']][mod['ITYPE(N)']==idMeas]
                rmse=RMSE(measF,modF)
                rrmse=rmse*100/measF['FOS'].mean()
                if k==4:
                    x_text=750
                else:
                    x_text=50
                ax.text(x_text,0.9*np.max(modF['FOS']),r'$RMSE_{mean}$='+str(round(rmse,2)),fontsize=8,bbox=dict(facecolor='white', edgecolor='none',alpha=1.))
                ax.text(x_text,0.6*np.max(modF['FOS']),r'$RRMSE_{mean}$='+str(round(rrmse,2))+' \%',fontsize=8,bbox=dict(facecolor='white', edgecolor='none',alpha=1))
            idMeas+=1
        pass
    for k in range(1,5):
        ax=fg.axes[k]
        ax.errorbar(df_meas.loc[:,'Time (hours)'],df_meas.loc[:,names[k-1]],yerr=df_std.loc[:,names[k-1]].values,fmt='o',markersize=4,capsize=5,color='k',label='_nolegend_') 
    
    ax=fg.axes[0]
    ax.legend(fontsize=10,ncol=3,loc=(0.05,-0.05),frameon=False)
    ax=fg.axes[-1]
    ax.legend(fontsize=10,framealpha=1,edgecolor=(1,1,1,1), loc=(0.03,0.03))
    # fg.savefig('predictive_'+treatment+'.pdf', format='pdf', bbox_inches='tight')
    pass



def plot_fluxes(sample,mean):
    it=0
    t_list=['LI','HI']
    fg=plt.figure(figsize=cm2inch(maxFigWidth,maxFigHeight),dpi=200)
    gs=gridspec.GridSpec(6,2)
    for treatment in t_list:
        if treatment=='HI':
            label_title='Irrigation Scenario: HI'
        else:
            label_title='Irrigation Scenario: LI'
            
        df_meas,df_std=load_measurements(treatment)
        model_name=treatment
        HYDRUS_path=os.getcwd()+'\\HYDRUS_models\\'+model_name
    
        #------------------CALIBRATED FRUIT PARAMS-------------------------------

        Krs = {'name':'Krs','x0':1e-4, 'bounds': np.array([5e-6,5e-4]), 'flags': np.array([1])} 
        Kcomp = {'name':'Kcomp', 'x0':1e-7, 'bounds': np.array([5e-8,5e-4]), 'flags': np.array([1])}
        hxMin = {'name':'hxMin','x0':-16000, 'bounds': np.array([-20000,-8000]), 'flags': np.array([0])} 
        if treatment=='LI':
            gamma = {'name':'gamma','x0':5.57, 'bounds': np.array([4,6]), 'flags': np.array([0])} 
            eta = {'name':'eta','x0':0.62, 'bounds': np.array([0.3,0.8]), 'flags': np.array([0])} 
        elif treatment=='HI':
            gamma = {'name':'gamma','x0':5.73, 'bounds': np.array([4,6]), 'flags': np.array([0])} 
            eta = {'name':'eta','x0':0.65, 'bounds': np.array([0.3,0.8]), 'flags': np.array([0])}         
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
        if treatment=='HI':
            model.th_tol=0.005
        else:
            model.th_tol=0.001
            
        label_plot=[r'$\textbf{(A)}$',r'$\textbf{(B)}$',r'$\textbf{(C)}$',r'$\textbf{(D)}$',r'$\textbf{(E)}$',r'$\textbf{(F)}$',
                    r'$\textbf{(G)}$',r'$\textbf{(H)}$',r'$\textbf{(I)}$',r'$\textbf{(J)}$',r'$\textbf{(K)}$',r'$\textbf{(L)}$']

        label_height=[-18,-0.3,0.2,0.01,0.01,1e-5]
            
        names_list=[r'$\Psi_{stem}$ (bar)',r'$U_{x}$ (gh$^{-1}$)', r'$U_{p}$ (gh$^{-1}$)', r'$U_{a}$ (gh$^{-1}$)', r'$U_{c}$ (gh$^{-1}$)',r'$U_{d}$ (gh$^{-1}$)']
        idMeas=40
        # 
        for i in range(6):
            ax=fg.add_subplot(gs[i,it],zorder=1)
            ax.set_xlim((0,2256))
            if i==0:
                ax.set_title(label_title)
                ax.set_ylim((-20,0))
            elif i==1:
                ax.set_ylim((-0.5,1.5))
            elif i==2:
                ax.set_ylim((0,2.5))
            elif i==3:
                ax.set_ylim((0,0.15))
            elif i==4:
                ax.set_ylim((0,0.15))
            else:
                ax.set_ylim((0,8e-5))
            if it==0:
                ax.set_ylabel(names_list[i],fontsize=8)
            else:
                ax.get_yaxis().set_visible(False)
            start = 0
            end=720
            width=end-start
            ymin,ymax=ax.get_ylim()
            rect = patches.Rectangle((start, ymin), width, abs(ymax-ymin), linewidth=0, edgecolor='k', facecolor='none',alpha=0.2,hatch='///')
            ax.add_patch(rect)
            if i==5 and it==0:
                ax.text(50,0.8*ymax,'SPIN-UP')
            # ax.text(2120,0.03, r'$\textbf{(A)}$')
            if i!=5:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Time (hours)', fontsize=12)
            ax.text(50,label_height[i], label_plot[i+it*6])
            idMeas+=1
        pass
        

        sample=np.vstack((sample,mean))
        for i in range(len(sample)):
            params=sample[i]
        
            thr,alfa,n,Ks,Krs,Kcomp,k,a,vm,delta,kst,beta_st,ksyn_so=params
            model.overwrite_params(params)
            model.execute()
        
            df_sugar=pd.read_csv(model.path_project+r'\SUGAR.OUT',sep='\s+',skiprows=1,header=0,index_col=0,usecols=[0,1])
            df_sugar_flux=pd.read_csv(model.path_project+r'\SUGAR_flux.OUT',sep='\s+',skiprows=1,header=0,index_col=0,usecols=[0,1,2,4,5,6])
                   
            if i==len(sample)-2:
                label='Modeled'
                color='b'
                alpha=0.3
            elif i==len(sample)-1:
                label='Mean'
                color='r'
                alpha=1
            else:
                label='_nolegend_'
                color='b'
                alpha=0.3
            
            ax=fg.axes[it*6]
            ax.plot(df_sugar.index,df_sugar.iloc[:,0]*9.80665E-4,color=color,alpha=alpha,label=label)
            
            for k in range(1,6):
                ax=fg.axes[it*6+k]
                ax.plot(df_sugar_flux.index,df_sugar_flux.iloc[:,k-1],color=color,alpha=alpha,label=label) 
                # if k==1:
                    # x1,x2,y1,y2=[1200,-0.25,1125,1]
                    # axins = ax.inset_axes([0.1,0.1,0.5,0.5],
                    #                       xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
                    # axins.plot(df_sugar_flux.index[1200:1224],df_sugar_flux.iloc[1200:1224,k-1])
                    # # print (df_sugar_flux.index[1200:1224])
                    # ax.indicate_inset_zoom(axins, edgecolor="black")

                        
        it+=1
    # fg.savefig('fluxes.pdf', format='pdf', bbox_inches='tight')
    pass

def plot_diurnal(sample,mean):
    it=0
    t_list=['LI','HI']
    fg=plt.figure(figsize=cm2inch(maxFigWidth*0.5,maxFigHeight*0.75),dpi=200)
    gs=gridspec.GridSpec(3,2,wspace=0.5)
    for treatment in t_list:
        if treatment=='HI':
            label_title='HI'
        else:
            label_title='LI'
        df_meas,df_std=load_measurements(treatment)
        model_name=treatment
        HYDRUS_path=os.getcwd()+'\\HYDRUS_models\\'+model_name
    
        #------------------CALIBRATED FRUIT PARAMS-------------------------------

        Krs = {'name':'Krs','x0':1e-4, 'bounds': np.array([5e-6,5e-4]), 'flags': np.array([1])} 
        Kcomp = {'name':'Kcomp', 'x0':1e-7, 'bounds': np.array([5e-8,5e-4]), 'flags': np.array([1])}
        hxMin = {'name':'hxMin','x0':-16000, 'bounds': np.array([-20000,-8000]), 'flags': np.array([0])} 
        if treatment=='LI':
            gamma = {'name':'gamma','x0':5.57, 'bounds': np.array([4,6]), 'flags': np.array([0])} 
            eta = {'name':'eta','x0':0.62, 'bounds': np.array([0.3,0.8]), 'flags': np.array([0])} 
        elif treatment=='HI':
            gamma = {'name':'gamma','x0':5.73, 'bounds': np.array([4,6]), 'flags': np.array([0])} 
            eta = {'name':'eta','x0':0.65, 'bounds': np.array([0.3,0.8]), 'flags': np.array([0])}         
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
        if treatment=='HI':
            model.th_tol=0.005
        else:
            model.th_tol=0.001
        
        ax_psi=fg.add_subplot(gs[0,it])
        ax_xyl=fg.add_subplot(gs[1,it])
        ax_tf=fg.add_subplot(gs[2,it])
        
        label_plot_psi=[r'$\textbf{(A)}$',r'$\textbf{(D)}$']
        label_plot_xyl=[r'$\textbf{(B)}$',r'$\textbf{(E)}$']
        label_plot_tf=[r'$\textbf{(C)}$',r'$\textbf{(F)}$']
        ax_psi.text(1250,-18, label_plot_psi[it])
        ax_xyl.text(1250,-0.8, label_plot_xyl[it])
        ax_tf.text(1250,0.4, label_plot_tf[it])
        
        sample=np.vstack((sample,mean))
        for i in range(len(sample)):
            params=sample[i]
        
            thr,alfa,n,Ks,Krs,Kcomp,k,a,vm,delta,kst,beta_st,ksyn_so=params
            model.overwrite_params(params)
            model.execute()
        
            df_sugar=pd.read_csv(model.path_project+r'\SUGAR.OUT',sep='\s+',skiprows=1,header=0,index_col=0,usecols=[0,1])
            df_sugar_flux=pd.read_csv(model.path_project+r'\SUGAR_flux.OUT',sep='\s+',skiprows=1,header=0,index_col=0)
            if i==len(sample)-2:
                label='Modeled'
                color='b'
                alpha=0.3
            elif i==len(sample)-1:
                label='Mean'
                color='r'
                alpha=1
            else:
                label='_nolegend_'
                color='b'
                alpha=0.3
            ax_psi.plot(df_sugar.index[1248:1272],df_sugar.iloc[1248:1272,0]*9.80665E-4,
                        color=color,alpha=alpha)
            ax_xyl.plot(df_sugar_flux.index[1248:1272],df_sugar_flux.iloc[1248:1272,0],
                        color=color,alpha=alpha)
            ax_tf.plot(df_sugar_flux.index[1248:1272],df_sugar_flux.iloc[1248:1272,2],
                        color=color,alpha=alpha)
    
            ax_xyl.axhline(0.0,color='0.5',linestyle='--')
            ax_xyl.xaxis.set_ticks([1248,1260,1272])
            ax_xyl.set_xticklabels([])
            ax_xyl.tick_params(axis='both', which='major', labelsize=10)
            ax_xyl.set_ylim((-1,1))
            ax_xyl.set_xlim((1248,1272))
            ax_xyl.yaxis.set_ticks([-1,0,1]) 
            if i==len(sample)-1 and it==0:
                ax_xyl.set_ylabel(r'$U_{x}$ (gh$^{-1}$)')
            if it==1:
                ax_xyl.set_yticklabels([])
            
            ax_psi.xaxis.set_ticks([1248,1260,1272])
            ax_psi.set_xticklabels([])
            ax_psi.tick_params(axis='both', which='major', labelsize=10)
            ax_psi.set_ylim((-20,0))
            ax_psi.set_xlim((1248,1272))
            ax_psi.yaxis.set_ticks([-20,-10,0])
            if i==len(sample)-1 and it==0:
                ax_psi.set_ylabel(r'$\Psi_{stem}$ (bar)')
            if it==1:
                ax_psi.set_yticklabels([])
            ax_psi.set_title(label_title)
            
            ax_tf.xaxis.set_ticks([1248,1260,1272])
            ax_tf.set_xticklabels(['00:00','12:00','24:00'])
            ax_tf.tick_params(axis='both', which='major', labelsize=10)
            ax_tf.set_ylim((0,0.5))
            ax_tf.set_xlim((1248,1272))
            ax_tf.yaxis.set_ticks([0,0.25,0.5])
            if i==len(sample)-1 and it==0:
                ax_tf.set_ylabel(r'$T_{f}$ (gh$^{-1}$)')
            if it==1:
                ax_tf.set_yticklabels([])
        it+=1
    # fg.savefig('diurnal.pdf', format='pdf', bbox_inches='tight')
    pass
        


chains=load_chains()
nSteps,nChains,nDims=chains.shape

L=int(0.1*nSteps)
cshort=chains[-L:,:,:]
flat=np.reshape(cshort,(L*nChains,nDims))

D=1
idx=idx=np.random.choice(np.arange(L*nChains),D,replace=False)
sample=flat[idx]
mean=np.mean(flat,axis=0)

p5,p95=np.percentile(flat,[5,95],axis=0)

import corner

def plot_marginal(flat,mean):
    flat_log=np.copy(flat)
    flat_log[:,4]=conv_to_log_cm_bar_h(flat_log[:,4])
    flat_log[:,5]=conv_to_log_cm_bar_h(flat_log[:,5])
    
    mean_log=np.copy(mean)
    mean_log[4]=conv_to_log_cm_bar_h(mean_log[4])
    mean_log[5]=conv_to_log_cm_bar_h(mean_log[5])
    
    CORNER_KWARGS = dict(
        smooth=0.9,
        smooth1d=0.9,
        label_kwargs=dict(fontsize=6),
        title_kwargs=dict(fontsize=16),
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        show_titles=False,
        max_n_ticks=3,
        labels=[r'$\theta_{r}$'+'\n'+'(-)', r'$\alpha$'+'\n'+r'(cm$^{-1}$)', r'$n$'+'\n'+'(-)',
                r'$K_{s}$'+'\n'+r'(cm h$^{-1}$)', r'$\log_{10}$'+'\n'+r'$k_{rs}^{max}{}\textsuperscript{\textdagger}$', r'$\log_{10}$'+'\n'+r'$k_{comp}^{max}{}\textsuperscript{\textdagger}$',
                r'$k$'+'\n'+'(-)', r'$a$'+'\n'+'(-)',r'$v_{m}$'+'\n'+r'($g g^{-1}h^{-1}$)', r'$\delta$'+'\n'+r'(h)', r'$\mu_{st}$'+'\n'+r'(h$^{-1}$)', 
                r'$\beta_{st}$'+'\n'+'(-)', r'$mu_{syn,so}$'+'\n'+r'(h$^{-1}$)'],
        labelpad=0.8,
        truths=mean_log,
        truth_color='r'
    )
    
    f=plt.figure(figsize=cm2inch(maxFigWidth,maxFigWidth),dpi=200)
    corner.corner(flat_log,bins=15,color='b',fig=f,**CORNER_KWARGS)
    for ax in f.get_axes():
        ax.tick_params(axis='both', labelsize=6)
    
    plt.figtext(-0.07,-0.11,r'${}\textsuperscript{\textdagger}$ cm bar$^{-1}$ h$^{-1}$',fontsize=6)
    # f.savefig('marginal.pdf', format='pdf', bbox_inches='tight')
    pass

def plot_chains(chains):
    fg=plt.figure(figsize=cm2inch(maxFigWidth,maxFigHeight),dpi=200)
    gs=gridspec.GridSpec(chains.shape[-1],1)
    labels=[r'$\theta_{r}$'+'\n'+'(-)', r'$\alpha$'+'\n'+r'(cm$^{-1}$)', r'$n$'+'\n'+'(-)',
            r'$K_{s}$'+'\n'+r'(cm h$^{-1}$)', r'$\log_{10}$'+'\n'+r'$k_{rs}^{max}{}\textsuperscript{\textdagger}$', r'$\log_{10}$'+'\n'+r'$k_{comp}^{max}{}\textsuperscript{\textdagger}$',
            r'$k$'+'\n'+'(-)', r'$a$'+'\n'+'(-)',r'$v_{m}$'+'\n'+r'($g g^{-1}h^{-1}$)', r'$\delta$'+'\n'+r'(h)', r'$\mu_{st}$'+'\n'+r'(h$^{-1}$)', 
            r'$\beta_{st}$'+'\n'+'(-)', r'$mu_{syn,so}$'+'\n'+r'(h$^{-1}$)']
    for i in range(chains.shape[-1]):
        ax=fg.add_subplot(gs[i,:])
        if i==4 or i==5:
            ax.plot(conv_to_log_cm_bar_h(chains[:,:,i]))
            if i==5:
                ymin=-5
                ymax=-0.
            if i==4:
                ymin=-5
                ymax=-1
        else:
            ax.plot(chains[:,:,i])
            ymin=chains[:,:,i].min()
            ymax=chains[:,:,i].max()
        ax.set_ylabel(labels[i],fontsize=8)
        ax.set_ylim((ymin-abs(0.5*ymin),ymax+abs(0.5*ymax)))
        ax.set_xlim((0,len(chains)))
        if i!=chains.shape[-1]-1:
            ax.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
            ax.tick_params(axis='y',labelsize=6)
        else:
            ax.tick_params(axis='both',labelsize=6)
            ax.set_xlabel(r'$Steps$ (-)')
            ax.text(0.1,-0.0015,r'${}\textsuperscript{\textdagger}$ cm bar$^{-1}$ h$^{-1}$')
    # fg.savefig('chains.pdf', format='pdf', bbox_inches='tight')
    pass


def Rhat(chains):
    N,M,D=chains.shape
    mu_chain=np.zeros((M,D))
    var_chain=np.zeros((M,D))
    for i in range(M):
        mu_chain[i,:]=chains[:,i,:].mean(axis=0)
        var_chain[i,:]=chains[:,i,:].var(axis=0,ddof=1)
    mu_all_chains=np.sum(mu_chain,axis=0)/M
    B=(N/(M-1))*np.sum((mu_chain-mu_all_chains)**2,axis=0)
    W=np.sum(var_chain,axis=0)/M
    R_hat=((N-1)*W/N+B/N)/W
    return R_hat**0.5

def plot_GR(chains):
    y=chains
    N = np.exp(np.linspace(np.log(100), np.log(y.shape[0]),20)).astype(int)
    GR= np.empty((len(N),y.shape[2]))
    for i,n in enumerate(N):
        GR[i,:]=Rhat(y[:n,:,:])
    plt.plot(N,GR)

def plot_acor(chains):
    y=chains
    N = np.exp(np.linspace(np.log(100), np.log(y.shape[0]),20)).astype(int)
    acor= np.empty((len(N),y.shape[2]))
    for i,n in enumerate(N):
        acor[i,:]=emcee.autocorr.integrated_time(y[:n,:,:],c=5,tol=0,quiet=True)
    plt.plot(N,acor)
    return N,acor


def plot_convergence(chains):
    
    names=[r'$\theta_{r}$', r'$\alpha$', r'$n$',
            r'$K_{s}$', r'$k_{rs}^{max}$', r'$k_{comp}^{max}$',
            r'$k$', r'$a$',r'$v_{m}$', r'$\delta$', r'$\mu_{st}$', 
            r'$\beta_{st}$', r'$mu_{syn,so}$']
    
    y=chains[:]
    N_intervals=20
    dy=int(len(y)/N_intervals)
    N=np.linspace(dy,len(y),N_intervals)
    GR= np.empty((len(N),y.shape[2]))
    acor= np.empty((len(N),y.shape[2]))
    for i,n in enumerate(N):
        GR[i,:]=Rhat(y[int(n/2):int(n),:,:])
        acor[i,:]=emcee.autocorr.integrated_time(y[:int(n),:,:],c=5,tol=0,quiet=True)
      
    fg=plt.figure(figsize=cm2inch(maxFigWidth,maxFigHeight*0.4),dpi=200)
    gs=gridspec.GridSpec(1,2,wspace=0.35)
    
    ax_IAT = fg.add_subplot(gs[0, 0])
    # Plot each parameter separately to control labels/colors
    for j in range(acor.shape[1]):
        ax_IAT.plot(N, acor[:, j], label=names[j])
    ax_IAT.set_ylabel(r'$\tau_{int}$')
    ax_IAT.set_xlim((0, 30000))
    ax_IAT.set_xticks([0, 10000, 20000, 30000])
    ax_IAT.set_xlabel(r'$Steps$ (-)')

    ax_GR = fg.add_subplot(gs[0, 1])
    for j in range(GR.shape[1]):
        ax_GR.plot(N, GR[:, j], label=names[j])
    ax_GR.set_ylabel(r'$\hat{R}$')
    ax_GR.axhline(1.2, linestyle='--', color='0.5')
    ax_GR.set_xlim((0, 30000))
    ax_GR.set_xticks([0, 10000, 20000, 30000])
    ax_GR.set_xlabel(r'$Steps$ (-)')
    
    fg.subplots_adjust(bottom=0.28)
    handles = ax_GR.get_lines()
    ncol = int(np.ceil(len(handles) / 2))
    fg.legend(handles, [l.get_label() for l in handles],
              loc='lower center', ncol=ncol, frameon=False, fontsize=8)
    
    # fg.savefig('MCMC_convergence.pdf', format='pdf', bbox_inches='tight')
    pass





