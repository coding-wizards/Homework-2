#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 18:50:24 2019

@author: Varunya Ilanghovan, Camilo Barrera
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

#Read the data
data=pd.read_csv('clusters.txt',names=['V1','V2'])


#Plot the data
f1=data['V1'].values
f2=data['V2'].values

X=np.array(list(zip(f1,f2)))
#plt.scatter(f1,f2,c='black',s=7)

class GMM:
    def __init__(self,X,number_sources,iterations):
        self.iterations=iterations
        self.number_sources=number_sources
        self.X=X
        self.mu=None
        self.pi=None
        self.cov=None
        self.XY=None
        
    #Function that runs for iterations"
    def run(self):
        self.reg_cov=1e-6*np.identity(len(self.X[0]))
        x,y=np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
        self.XY=np.array([x.flatten(),y.flatten()]).T
        
        #Set the initial mu, covariance and pi values
        self.mu=np.random.randint(min(self.X[:,0]),max(self.X[:,0]),size=(self.number_sources,len(self.X[0])))
        self.cov=np.zeros((self.number_sources,len(X[0]),len(X[0])))
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim],5)
        self.pi=np.ones(self.number_sources)/self.number_sources
        log_likelihoods=[]
        
        #Plot the initial state
        fig=plt.figure(figsize=(10,10))
        ax0=fig.add_subplot(111)
        ax0.scatter(self.X[:,0],self.X[:,1])
        ax0.set_title('Iinitial state')
        for m,c in zip(self.mu,self.cov):
            c += self.reg_cov
            multi_normal=multivariate_normal(mean=m,cov=c)
            ax0.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]),multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
            ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
        
        for i in range(self.iterations):
            #E step
            r_ic=np.zeros((len(self.X),len(self.cov)))
            for m,co,p,r in zip(self.mu,self.cov,self.pi,range(len(r_ic[0]))):
                co+=self.reg_cov
                mn=multivariate_normal(mean=m,cov=co)
                r_ic[:,r]=p*mn.pdf(self.X)/np.sum([pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(X) for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.cov+self.reg_cov)],axis=0)
            
        #M Step
        #Calculate the new mean vector and new covariance matrices
        
        self.mu=[]
        self.cov=[]
        self.pi=[]
        log_likelihood=[]
        for c in range(len(r_ic[0])):
            m_c=np.sum(r_ic[:,c],axis=0)
            mu_c=(1/m_c)*np.sum(self.X*r_ic[:,c].reshape(len(self.X),1),axis=0)
            self.mu.append(mu_c)
            #Calculate the covariance matrix per source based on the new mean
            self.cov.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.X),1)*(self.X-mu_c)).T,(self.X-mu_c)))+self.reg_cov)
            #Calculate pi_new 
            self.pi.append(m_c/np.sum(r_ic))
        """    
        #Log likelihood
        log_likelihoods.append(np.log(np.sum([k*multivariate_normal(self.mu[i],self.cov[i],self.cov[j]).pdf(X) for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])))
        
        fig2=plt.figure(figsize=(10,10))
        ax1=fig2.add_subplot(111)
        ax1.set_title('Log-Likelihood')
        ax1.plot(range(0,self.iterations,1),log_likelihoods)
        plt.show()
        """
        print('means: ','\n',self.mu)
        print('amplitud: ','\n',self.pi)
        print('covariance: ','\n',self.cov)
        
GMM=GMM(X,3,50)
GMM.run()

"""
Applying Gaussian Models using the scikit-learn library
"""

gmm=GaussianMixture(n_components=3)
gmm.fit(X)

print('means scikit: ','\n',gmm.means_)
print('\n')
print('covariances scikit: ','\n',gmm.covariances_)

X1,X2=np.meshgrid(np.linspace(-10,15),np.linspace(-10,15))
XX=np.array([X1.ravel(),X2.ravel()]).T
Z=gmm.score_samples(XX)
Z=Z.reshape((50,50))
"""
plt.contour(X1,X2,Z)
plt.scatter(X[:,0],X[:,1])
plt.show
"""
