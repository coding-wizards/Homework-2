# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:13:40 2019

@author: Varunya Ilanghovan, Camilo Barrera
"""
#Load important libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.cluster import KMeans

#Read the data
data=pd.read_csv('clusters.txt',names=['V1','V2'])
#print (data)

#Plot the data
f1=data['V1'].values
f2=data['V2'].values

X=np.array(list(zip(f1,f2)))
plt.scatter(f1,f2,c='black',s=7)

#Function to calculate the Euclidean distance
def dist(a,b,ax=1):
    return np.linalg.norm(a-b,axis=ax)

#Set the number of clusters to k=3
k=3

#Create the x and y coordinates for the centroids
c_x=np.random.randint(0,np.max(X),size=k)
c_y=np.random.randint(0,np.max(X),size=k)
c=np.array(list(zip(c_x,c_y)),dtype=np.float32)
print(c)

#Create plot with the centroids
plt.scatter(f1,f2,c='#050505',s=7)
plt.scatter(c_x,c_y,marker='*',s=200,c='g')

#Variable to store the old values of the centroids
c_old=np.zeros(c.shape)
#Cluster labels (0,1,2)
clusters=np.zeros(len(X))
#Error function -Distance between new centroids and old centroids
error=dist(c,c_old,None)
#Loop to assign the data points to the clusters and calculate new centroids
while error !=0:
    #Assign each data point to its closest centroid
    for i in range(len(X)):
        distances=dist(X[i],c)
        cluster=np.argmin(distances)
        clusters[i]=cluster
    #Store the old centroid values
    c_old=deepcopy(c)
    #Find the new centroids
    for i in range(k):
        points=[X[j]for j in range(len(X))if clusters[j]==i]
        c[i]=np.mean(points,axis=0)
    error=dist(c,c_old,None)
    
colors=['r','g','b','y','c','m']
fig, ax=plt.subplots()
for i in range(k):
    points=np.array([X[j]for j in range(len(X))if clusters[j]==i])
    ax.scatter(points[:,0],points[:,1],s=7,c=colors[i])
    
ax.scatter(c[:,0],c[:,1],marker='*',s=200,c='#050505')



#####################################################################
#Implementation using an SciKit library (KMeans)
#Set the number of clusters
kmeans=KMeans(n_clusters=3)
#Fit the imput data
kmeans=kmeans.fit(X)
#Get the cluster labels
labels=kmeans.predict(X)
#Centroid values
centroids=kmeans.cluster_centers_

#Compare our approach with scikit centroids
print("our implementation: ",c)
print("scikit implementation: ",centroids)