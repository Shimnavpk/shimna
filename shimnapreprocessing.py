#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt


# In[2]:


#Load the data
df=pd.read_csv("C:\\Users\\Noel\\Downloads\\titanic_dataset (1).csv", index_col ='PassengerId')
df.head()


# In[3]:


df.info()


# In[4]:


df.columns


# In[5]:


df.shape


# In[6]:


df=pd.get_dummies(df)
df.head()


# In[7]:


df.isnull().any().any()


# In[8]:


df.isnull().sum()


# In[9]:


df.dtypes


# In[10]:



freq_graph = df.select_dtypes(include='float')
freq_graph.hist(figsize = [20,15]);


# In[11]:


df['Age']=df['Age'].fillna(df['Age'].median())
df['Age'].isna().sum()


# In[12]:


df=df.drop('Age',axis=1)
df.head()


# In[13]:


df.isnull().any().any()


# In[14]:


# Position of the Outlier
print(np.where(df.select_dtypes(include='float')>10))


# In[15]:


#Detecting the outliers
import seaborn as sns
sns.boxplot(df['Fare'])


# In[16]:


Q1=np.percentile(df['Fare'],25)
Q1


# In[17]:


Q2=np.percentile(df['Fare'],50)
Q2


# In[18]:


Q3=np.percentile(df['Fare'],75)
Q3


# In[19]:


IQR=Q3-Q1
IQR


# In[21]:


low_limit=Q1-1.5*IQR
upr_lim=Q3+1.5*IQR
low_limit


# In[22]:


upr_lim


# In[23]:


outlier = []
for x in df['Fare']:
    if((x>upr_lim)or(x<low_limit)):
        outlier.append(x)
outlier


# In[24]:


ind = df['Fare']>upr_lim
df.loc[ind].index


# In[25]:


df.drop([ 2,  28,  32,  35,  53,  62,  63,  73,  89, 103,
            
            793, 803, 821, 830, 836, 847, 850, 857, 864, 880],inplace=True)
df.shape


# In[26]:


ind = df['Fare']>upr_lim
df = df.loc[:, df.notnull().any(axis = 0)]
print (df)


# In[27]:


sns.boxplot(df['Fare'])


# In[28]:


x=df.drop('Survived',axis=1)
y=df[['Survived']]


# In[29]:


from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler().fit(df[['Fare']])
df_copy = df.copy(deep=True)
df_copy['Fare_minmax'] =min_max.transform(df_copy[['Fare']])
print(df_copy.head())


# In[30]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state =42,test_size=0.2)


# In[31]:


from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression()
LR_model = LR_model.fit(x_train,y_train)


# In[32]:


score_LR = LR_model.score(x_test,y_test)


# In[33]:


score_LR 


# #### knn-k-nearest neighbours

# In[34]:



from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train,y_train,)
print("Accuracy :",model.score( x_test,y_test)*100)


# #### SVM-linear

# In[35]:


from sklearn.svm import SVC
svm_cls = SVC(kernel='linear')
svm_cls = svm_cls.fit(x_train,y_train)
y_pred_svm = svm_cls.predict(x_test)


# In[36]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[37]:


confusion_matrix(y_test,y_pred_svm)


# In[38]:


accuracy_score(y_test,y_pred_svm)


# #### KMeans clustering

# In[41]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,13):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit((df)) 
    wcss.append(kmeans.inertia_)
plt.plot(range(1,13),wcss)
plt.title('The Elbow Method') 
plt.xlabel('No.of clusters')
plt.ylabel('wcss value')
plt.show()
    
    


# In[43]:


kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(df)
y_kmeans


# #### Hierarchial Clustering

# In[44]:


import scipy.cluster.hierarchy as sch


# In[45]:


dendrogram = sch.dendrogram(sch.linkage(df,method='ward'))
plt.title('dendrogram')
plt.xlabel('Datapoints')
plt.ylabel('Eucleidean distance')
plt.show()


# #### Kfold cross validation

# In[23]:


from sklearn.model_selection import KFold
kfold_validator=KFold(10)
for train_index,test_index in kfold_validator.split(x,y):
    print('Training Index:',train_index)
    print('Validation Index:',test_index)
    


# #### Stratified k-fold cross validation 

# In[26]:


from sklearn.model_selection import cross_val_score
cv_result=cross_val_score(LR_model ,x,y,cv=kfold_validator)


# In[28]:


cv_result


# In[29]:


np.mean(cv_result)


# In[ ]:




