#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# In[43]:


#Load the data
df=pd.read_csv("C:\\Users\\Noel\\Downloads\\\Wine_clust.csv")
df.head()


# In[44]:


df.describe()


# In[45]:


df.info()


# In[46]:


df.isnull().any().any()


# In[47]:


df.columns


# In[48]:


df.isna().sum()


# In[49]:


df1=df.drop(['Magnesium','Proline'],axis=1)
df1.head(3)


# In[51]:


df2=df1.iloc[:,[3,4]].values


# In[52]:


df2.shape


# In[53]:


df1.columns


# ### KMeans clustering

# In[75]:


wcss=[]
for i in range(1,13):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit((df1)) 
    wcss.append(kmeans.inertia_)
plt.plot(range(1,13),wcss)
plt.title('The Elbow Method') 
plt.xlabel('No.of clusters')
plt.ylabel('wcss value')
plt.show()
    
    


# In[54]:


kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(df2)
y_kmeans


# In[57]:


plt.scatter(df2[y_kmeans==0,0],df2[y_kmeans==0,1],s=100,c='purple',label='cluster1')


# ### Hierarchial Clustering

# In[58]:


import scipy.cluster.hierarchy as sch


# In[59]:


dendrogram = sch.dendrogram(sch.linkage(df2,method='ward'))
plt.title('dendrogram')
plt.xlabel('Datapoints')
plt.ylabel('Eucleidean distance')
plt.show()


# ### AgglomerativeClustering

# In[62]:


from sklearn.cluster import AgglomerativeClustering
ahc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_ahc=ahc.fit_predict(df2)
y_ahc


# In[63]:


from sklearn.metrics import silhouette_score
sil_ahc = silhouette_score(df2,y_ahc)
sil_ahc 


# ### Principal Component Analysis

# In[64]:


from sklearn.preprocessing import StandardScaler


# In[65]:


scaler=StandardScaler()
scaled_data=scaler.fit_transform(df2)
scaled_data


# In[66]:


from sklearn.decomposition import PCA


# In[67]:


pca =PCA(n_components=0.97)
pca.fit(scaled_data)


# In[68]:


x_pca = pca.transform(scaled_data)
x_pca


# In[69]:


x_pca.shape


# In[70]:


np.cumsum(pca.explained_variance_ratio_)


# In[71]:


pca.components_


# ### dbscan Clustering

# In[39]:


# extracting the above mentioned columns
x = df.loc[:, ['Alcohol','Color_Intensity']].values


# In[40]:


from sklearn.cluster import DBSCAN
# cluster the data into five clusters
dbscan = DBSCAN(eps = 8, min_samples = 4).fit(x) # fitting the model
labels = dbscan.labels_ # getting the labels


# In[74]:


# Plot the clusters
plt.scatter(x[:, 0], x[:,1], c = labels, cmap='summer') # plotting the clusters
plt.xlabel("Alcohol") # X-axis label
plt.ylabel("Color_Intensity") # Y-axis label
plt.show() # showing the plot


# In[ ]:




