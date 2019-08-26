# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:31:00 2019
bio.cluster.clustercentriods Test
@author: Administrator
"""

from sklearn.preprocessing import MinMaxScaler
from Bio.Cluster import kcluster
from Bio.Cluster import clustercentroids
import pandas as pd

df_full = pd.read_csv('../data/FCM_testdata.csv')
print(df_full.head(5))
print('size of df_full with real_labels:',df_full.shape)
columns = list(df_full.columns)
features = columns[:len(columns)-1]
df = df_full[features]
print(df.head(5))
print('size of df without real_labels:',df.shape)
minMax = MinMaxScaler()
print(minMax)
dataset=minMax.fit_transform(df)
print(dataset[0:10,:])

clusterid1, error, nfound = kcluster(dataset, nclusters=3, dist='e',npass=100)
cdata, cmask = clustercentroids(dataset,clusterid=clusterid1)