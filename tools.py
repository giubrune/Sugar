# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import time
import subprocess as sub
from shutil import copy
from copy import deepcopy
import os
import re

class Model(object):
    def __init__(self,path_project=None,additional_params_list=None,
                 execution_time=None):        

        self.path_project=path_project #path to the HYDRUS files
        self.exec_time=execution_time
        self.create_LEVEL_01()
        self.gen_info=self.read_general_info()  
        self.HYD_info=self.read_HYDRUS_info()
        self.model_type=self.gen_info['Model']
        self.observations=self.read_observations()
        self.nObs=len(self.observations)
        self.x0,self.lb,self.ub,self.flags=self.read_parameter_bounds()
        self.add_par=additional_params_list
        self.update_model_par(self.add_par)
        self.tMax=self.read_tMax()
        self.nOpt=0
        self.optPar=[]
        self.th_tol=0.001

        for key, value in self.flags.items():
            s=int(sum(value))
            self.nOpt+=s
            if s>0.:
                self.optPar.append(key)
                
    def update_model_par(self,par_list):
        if par_list is not None:
            for i in range(len(par_list)):
                self.flags[par_list[i]['name']]=par_list[i]['flags']
                try:
                    self.x0[par_list[i]['name']]=np.array([float(par_list[i]['x0'])])
                    self.lb[par_list[i]['name']]=np.array([float(par_list[i]['bounds'][0])])
                    self.ub[par_list[i]['name']]=np.array([float(par_list[i]['bounds'][1])])
                except:
                    self.x0[par_list[i]['name']]=par_list[i]['x0']
                    self.lb[par_list[i]['name']]=par_list[i]['bounds'][:,0]
                    self.ub[par_list[i]['name']]=par_list[i]['bounds'][:,1]
                
    def read_tMax(self):
        with open(self.path_project+'\SELECTOR.IN','r') as file:
            f=file.readlines()               
            for l in range(len(f)):
                line=f[l].split()
                if ('tMax' in line) and (len(line)==2):
                    tMax=float(f[l+1].split()[1])
                    break
        return tMax
    
    def convert_pars_to_dict(self,params):
        params=np.round(params,10)
        d=deepcopy(self.x0)
        f=self.flags
        j=0
        for k,v in d.items():
            for i in range(len(v)):
                if f[k][i]==1:
                    d[k][i]=params[j]
                    j+=1
        return d

    
    def convert_dict_to_pars(self,d):
        params=np.zeros(self.nOpt)
        j=0
        for k,v in self.flags.items():
            for i in range(len(v)):
                if v[i]==1:
                    params[j]=d[k][i]
                    j+=1                
        return params
    
    def create_LEVEL_01(self):
        with open(self.path_project+'/LEVEL_01.DIR','w') as file:
            file.write(self.path_project)
            pass  
        
    def read_general_info(self):        
        keys=['NOBB','iWeight','lWatF','lChemF','NMat','lTempF','Model']
        dicts=dict(zip(keys,[0]*len(keys)))
        with open(self.path_project+'\FIT.IN') as file:
            f=file.read().splitlines()
            for key in keys:
                for l in range(len(f)):
                    for k in range(len(f[l].split())):
                        if key==f[l].split()[k]:
                            dicts[key]=f[l+1].split()[k]        
        return dicts
    
    def read_HYDRUS_info(self): 
        keys=['HYDRUS_Version','WaterFlow','SoluteTransport','Unsatchem','HP1',
                  'HeatTransport','EquilibriumAdsorption','MobileImmobile','RootWaterUptake',
                  'RootGrowth','MaterialNumbers','SubregionNumbers','SpaceUnit','TimeUnit',
                  'PrintTimes','NumberOfSolutes','InitialCondition','NumberOfNodes','ProfileDepth',
                  'ObservationNodes','GridVisible','SnapToGrid','ProfileWidth','LeftMargin','GridOrgX',
                  'GridOrgY','GridDX','GridDY']
        dicts=dict(zip(keys,[0]*len(keys)))          
        with open(self.path_project+'\HYDRUS1D.DAT') as file:
            f=file.read().splitlines()
            for key in keys:
                for l in range(len(f)):
                    if f[l].startswith(key):
                        L=len(key)
                        try:
                            dicts[key]=float(f[l][L+1:])
                        except:
                            dicts[key]=f[l][L+1:]
                        break
        return dicts
    
    def read_obsnode_pos(self):
        nNodes=int(self.HYD_info['NumberOfNodes'])
        with open(self.path_project+'\PROFILE.DAT') as file:
            f=file.read().splitlines()
            obs_pos=np.array([int(i) for i in f[-1].split()])
            for l in range(len(f)):
                if 'Axz' in f[l]:
                    l_start=l+1
                    break
            output=np.loadtxt(f[:nNodes],skiprows=l_start)
        return np.rint(output[[obs_pos-1],1][0])                                   
    
    def read_observations(self):
        keys=['HO(N)', 'FOS', 'ITYPE(N)', 'POS', 'WTS']
        NObs=int(self.gen_info['NOBB'])
        df_obs=pd.DataFrame(index=np.arange(NObs),columns=keys)
        with open(self.path_project+'\FIT.IN') as file:
            f=file.read().splitlines()
            for l in range(len(f)):
                if keys[0] in f[l]:
                    break
        with open(self.path_project+'\FIT.IN') as file:
            f=file.read().splitlines()
            row=0
            for ll in range(l+1,l+NObs+1):
                df_obs.loc[row,:]=[float(i) for i in f[ll].split()]
                row+=1
        df_obs=df_obs.astype(float)
        type_list=list(set(df_obs['ITYPE(N)']))       
        for t in range(len(type_list)):
            iWeights=1.
            df_obs.loc[df_obs['ITYPE(N)']==type_list[t],'WTS']=df_obs['WTS']/iWeights
        return df_obs
    
    def read_parameter_bounds(self):
        keys_par=['thr','ths', 'Alfa', 'n', 'Ks', 'l']
        lb=np.zeros((1,len(keys_par)))
        ub=np.zeros((1,len(keys_par)))
        fgs=np.zeros((1,len(keys_par)))
        x0=np.zeros((1,len(keys_par)))
        with open(self.path_project+'\FIT.IN') as file:
            f=file.read().splitlines()
            l_start=0
            for l in range(l_start,len(f)):
                if "thr" in f[l]:
                    x0[0,:]=[float(i) for i in f[l+1].split()]
                    fgs[0,:]=[float(i) for i in f[l+2].split()]
                    lb[0,:]=[float(i) for i in f[l+3].split()]
                    ub[0,:]=[float(i) for i in f[l+4].split()]                        
                    l_start=l+4
                    break                        
        upper_bounds=dict(zip(keys_par,ub.T))
        lower_bounds=dict(zip(keys_par,lb.T))
        flags=dict(zip(keys_par,fgs.T))
        x0=dict(zip(keys_par,x0.T))
        return x0,lower_bounds,upper_bounds,flags
 
    def check_convergence(self):
        try:
            with open(self.path_project+r'\Run_Inf.out','r') as file:
                f=file.read().splitlines()
                if float(f[-2].split()[1])==self.tMax:
                    converged=True
                else:
                    converged=False
        except:
            converged=False
        return converged
    
    def read_model_output(self):
        converged=self.check_convergence()
        df_sim=self.observations.copy()
        simulations=df_sim.values
        if converged:
            try:
                filelist,flags=self.load_HYDRUS_output()            
                lObs=len(self.observations)
                obs_array=self.observations.values
                for j in range(len(filelist)):
                    flag=flags[j]
                    f=filelist[j].read().splitlines()
                    flag_read=False
                    for i in range(lObs):
                        code=int(obs_array[i,2])
                        pos=int(obs_array[i,3]) 
                        if flag=='Obs_Nod':
                            l_start=11
                            if flag_read==False:
                                output=np.loadtxt(f[:-1],skiprows=l_start)
                            if len(output)!=0:
                                flag_read=True
                            time=obs_array[i,0]
                            idx=np.where(output[:,0]==time)[0]
                        elif flag=='Sugar':
                            l_start=2
                            if flag_read==False:
                                output=np.loadtxt(f[:-1],skiprows=l_start)
                            if len(output)!=0:
                                flag_read=True
                            time=obs_array[i,0]
                            idx=np.where(output[:,0]==time)[0]                            
                            
                        if len(idx)>1:
                            idx=idx[-1]
                        if flag=='Obs_Nod':
                            idx_col=(pos-1)*3
                            simulations[i,1]=output[idx,2+idx_col]
                        elif flag=='Sugar':
                            if code==40:
                                idx_col=2
                                simulations[i,1]=output[idx,idx_col]
                            elif code==41:
                                idx_col=3
                                simulations[i,1]=output[idx,idx_col]
                            elif code==42:
                                idx_col=4
                                simulations[i,1]=output[idx,idx_col]
                            elif code==43:
                                idx_col=5
                                simulations[i,1]=output[idx,idx_col] 
                        
            except:
                simulations[:,1]=np.inf                    
        else:
            simulations[:,1]=np.inf
        if np.isnan(np.sum(simulations[:,1])):
            simulations[:,1]=np.inf
        df_sim.loc[:,:]=simulations
        return df_sim 
          
    def load_HYDRUS_output(self):
        code_list=list(set(self.observations['ITYPE(N)']))
        filelist=[]
        flags=[]
        for code in code_list: 
            code=int(code)
            obs_code=self.observations[self.observations['ITYPE(N)']==code]
            pos_list=list(set(obs_code['POS']))          
            for pos in pos_list:
                if code==2:
                    filename=self.path_project+'\Obs_Node.out'
                    flag='Obs_Nod'
                elif (code==40) or (code==41) or (code==42) or (code==43):
                    filename=self.path_project+r'\SUGAR.out'
                    flag='Sugar'
                file_obs=open(filename,'r')
                filelist.append(file_obs)
                flags.append(flag)  
        return filelist,flags
    
    def overwrite_params(self,params):
        params=self.convert_pars_to_dict(params) 
        hCritA=self.inv_VGM(params['thr'][0], params['ths'][0], params['Alfa'][0], params['n'][0])           
        self.overwrite_hCrit(hCritA)

        with open(self.path_project+'\SELECTOR.IN','r') as file:
            f=file.readlines()
            for key,value in params.items():                   
                for l in range(len(f)):
                    line=f[l].split()

                    if (key in line) and ('ths' in line): #line containing thR,thS,a,n,Ks,l
                        idx=line.index(key)
                        for j in range(1,len(value)+1):
                            ll=f[l+j].split()
                            ll[idx]=str(round(value[j-1],4))
                            f[l+j]=' '.join(map(str,ll))+' \n'
                        break
                    if key in line:
                        idx=line.index(key)
                        for j in range(1,len(value)+1):
                            ll=f[l+j].split()
                            ll[idx]=str(round(value[j-1],8))
                            f[l+j]=' '.join(map(str,ll))+' \n'
                        break
        with open(self.path_project+'\SELECTOR.IN','w') as file:
            for line in f:
                file.write(line)
                pass
            pass
        
        with open(self.path_project+'\SUGAR.IN','r') as fileS:
            fs=fileS.readlines()
            for key,value in params.items():                   
                for l in range(len(fs)):
                    line=fs[l].split()
                    if key in line: 
                        idx=line.index(key)
                        for j in range(1,len(value)+1):
                            ll=fs[l+j].split()
                            ll[idx]=str(value[j-1])
                            fs[l+j]=' '.join(map(str,ll))+' \n'
                        break
        with open(self.path_project+'\SUGAR.IN','w') as fileS:
            for line in fs:
                fileS.write(line)
                pass
            pass
        pass
    pass 
    
    def wait_timeout(self,Proc,seconds):
        starts=time.time()
        end=starts+seconds
        interval=min(seconds/1000.0,.10)        
        while True:
            result=Proc.poll()
            if result is not None:
                return result
            if time.time() >=end:
                Proc.terminate()
                Proc.wait()
            time.sleep(interval)
    
    def execute(self):
        Proc=sub.Popen(self.path_project+'/H1D_Sugar.exe',cwd=self.path_project,stdout=sub.PIPE)  
        self.wait_timeout(Proc,self.exec_time)
        pass
  
    
    def overwrite_hCrit(self,hCritA):
        if hCritA>100000:
            hCritA=100000.
        if hCritA<300:
            hCritA=300
        flag=False
        for it in range(2):
            try:
                with open(self.path_project+'\ATMOSPH.IN','r') as file:
                    f=file.readlines()
                    lstart=False
                    for l in range(len(f)):
                        line=f[l].split()
                        if lstart:
                            L=f[l].split()
                            L[4]=str(hCritA)
                            string=' '.join(map(str,L))+' \n'
                            f[l]=string
                        if ('hCritA' in line):
                            lstart=True
                with open(self.path_project+'\ATMOSPH.IN','w') as file:
                    for line in f:
                        file.write(line)
                        pass
                    pass
                pass
                flag=True
            except:
                time.sleep(0.01)
                pass
            if flag==True:
                break

    def inv_VGM(self,tr,ts,a,n):
        th=tr+self.th_tol
        Se=(th-tr)/(ts-tr)
        psi=(1./(a*(Se**(1./(1.-1./n)))**(1./n)))-1./a
        if psi>10000000.:
            psi=10000000.
        return int(psi)
