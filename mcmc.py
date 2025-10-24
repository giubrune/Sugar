# -*- coding: utf-8 -*-
"""

@author: giuseppe brunetti
"""

import os
import shutil
from tqdm import trange
import numpy as np

            
class AIES(object):
    def __init__(self, model=None, additional_params=None):
    
        self.model=model    #instance of the model
        self.lb, self.ub=self.convert_bounds() #convert dictionary of bounds to arrays
        self.add_par=additional_params
        self.MeasID=list(set(self.model.observations['ITYPE(N)'])) #ID of differenet measurements
        self.MeasID.sort()
        self.nMeasType=len(self.MeasID) #number of different measurements       
        self.unpack_additional_params(additional_params) #unpacks additional parameters that cannot be declared in HYDRUS GUI
        self.nDims=len(self.lb) #dimensionality of the inverse problem
        
        
    def spherical_walkers_init(self,x0,std,nChains):
        start=np.random.multivariate_normal(x0,np.diag(std)**2,size=nChains)
        return start    
       
    def unpack_additional_params(self,add_par):
        assert len(add_par)>=self.nMeasType, 'A Gaussian Likelihood requires sigma for each measurement type'
        assert ('sigma1' in add_par)
        self.sigmas=np.zeros(self.nMeasType)
        it=0
        for k,v in add_par.items():
            if v['flag']==1:
                self.lb=np.append(self.lb,v['bounds'][0])
                self.ub=np.append(self.ub,v['bounds'][1])
            else:
                self.sigmas[it]=v['x0']
            it+=1

    def logprior(self,params):
        if np.any(params < self.lb) or np.any(params > self.ub):
            return -np.inf
        else:
            return 0           
    
    def loglike(self,params,mod):
        if all(self.sigmas==0):
            sigmas=params[-self.nMeasType:] #used in the Gaussian likelihood. Sigma is placed at the end of the parameters list.
            params=params[:-self.nMeasType] #to exclude the last nMeasType values representing sigma in the gaussian
        else:
            sigmas=self.sigmas
            params=params
        mod.overwrite_params(params) #Overwrites parameters and hCritA
        mod.execute() #executes HYDRUS
        df=mod.read_model_output() #reads the model output
        resid=(mod.observations-df)**2 #calculates the squared residuals
        resid['FOS']=resid['FOS']*self.model.observations['WTS'] #multiplies residuals by weights declared in the hydrus GUI.
        SSQ=resid['FOS'].sum() #just to check if execution was convergent
        if SSQ==np.inf:
            flag=1 #the model didn't converge
            like=-np.inf #we attribute a negative infinite value to the likelihood
        else:
            flag=0 #the model converged
            like=0 #we initialize the like and loop through different measurement types to calculate the aggreated sum of loglikelihoods
            for i in range(self.nMeasType):
                SSQ=resid['FOS'][df['ITYPE(N)']==self.MeasID[i]].sum() #sum of squared residuals for the ith measurement set
                N=resid['FOS'][df['ITYPE(N)']==self.MeasID[i]].count() #number of observations in the ith measurement set
                l=-0.5*N*np.log(2*np.pi)-0.5*N*np.log(sigmas[i]**2)-(0.5/sigmas[i]**2)*SSQ
                like+=l #calculate the loglikelihood
        return like,flag

    def logprob(self,params,mod,results,idx):
        lp=self.logprior(params) #calculates the logprior probability
        if not np.isfinite(lp): #if the prior is not finite we can skip the calculation of loglike
            flag=0 #the model is still assumed to have converged
            results.append([idx,-np.inf,flag]) #we append -np.inf
        else:
            l,flag=self.loglike(params,mod) #calculate the loglikelihood
            results.append([idx,lp+l,flag]) #appends the final logprob
        pass
    
    def convert_bounds(self):
        lb=self.model.convert_dict_to_pars(self.model.lb)
        ub=self.model.convert_dict_to_pars(self.model.ub)
        return lb,ub
    
    def append_output(self,chains_coords,logprob):
        for d in range(self.nDims):
            filename=os.path.join(self.output_path,'chains_par_'+str(d)+'.txt')
            with open(filename,'a') as f:
                f.write(" ".join(map("{:.4E}".format, chains_coords[:,d]))+' \n')
                f.close()
        filename=os.path.join(self.output_path,'logprob.txt')
        with open(filename,'a') as f:
            f.write(" ".join(map("{:.4E}".format, logprob))+' \n')
            f.close()
    
    def stretch_move(self,it):
        # carry forward previous iteratio
        self.chains[it, :, :] = self.chains[it - 1, :, :]
        self.lnlike[it, :]    = self.lnlike[it - 1, :]
        Z = np.empty(self.nChains,dtype=float)
        Y = np.empty_like(self.chains[it,:,:])          # (nChains, d)
        post_Y = np.empty(self.nChains,dtype=float)
        for i in range(self.nChains):
            comp_ensemble = np.delete(np.arange(self.nChains), i)
            j = np.random.choice(comp_ensemble)        
            x_i = self.chains[it, i, :]
            x_j = self.chains[it, j, :]
            Z[i] = ((self.a - 1.0) * np.random.rand() + 1.0) ** 2 / self.a
            Y[i, :] = x_j + Z[i] * (x_i - x_j)
        
            r=[]
            self.logprob(Y[i,:], self.model, r, 0)
            post_Y[i] = r[0][1]
            log_accept = (self.nDims - 1) * np.log(Z[i]) + post_Y[i] - self.lnlike[it, i]
            if np.log(np.random.rand()) < log_accept:
                self.chains[it, i, :] = Y[i, :]
                self.lnlike[it, i]    = post_Y[i]
            # else: keep carried-forward state already in self.chains[it, i, :]            

    
    def run_mcmc(self,nChains,a,nSteps,start,Output_folder_name=None):
        self.a=a
        self.nChains=int(nChains)
        self.chains=np.zeros((nSteps,self.nChains,self.nDims))                
        code_path=os.getcwd()
        self.output_path=code_path+r"\\"+Output_folder_name+'_output'        
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.mkdir(self.output_path)
        for j in range(self.nDims):
            filename=os.path.join(self.output_path,'chains_par_'+str(j)+'.txt')
            f=open(filename,'x')
            f.close()
        filename=os.path.join(self.output_path,'logprob.txt')
        f=open(filename,'x')
        f.close()
                
        self.chains[0,:,:]=start
        self.lnlike=np.zeros((nSteps,self.nChains))        
        print ('Initializing '+str(self.nChains)+' Chains')         
        for i in range(self.nChains):
            r=[]
            self.logprob(self.chains[0,i,:], self.model, r, 0)
            self.lnlike[0,i]=r[0][1]
        self.append_output(self.chains[0,:,:],self.lnlike[0,:])
        
        t = trange(1,nSteps, desc='AIES', leave=True)        
        for it in t:            
            self.stretch_move(it)        
            self.append_output(self.chains[it,:,:],self.lnlike[it,:])
        pass
    

    
