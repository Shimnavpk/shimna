#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

 #Load the data
df=pd.read_csv("C:\\Users\\Noel\\Downloads\\titanic_dataset.csv", index_col ='PassengerId')
df.head()


# In[8]:


df.shape


# In[10]:


df.info()


# In[7]:


df.describe()


# In[11]:


# find missing values
df.isna().sum()


# In[7]:


# plot missing values heatmap
sns.heatmap(df.isna(), cbar=False, cmap='cividis')
plt.title('Missing values in each columns')
plt.show()


# In[12]:


df.columns


# In[16]:


df.dropna(inplace = True)
df


# In[39]:



num_cols = df[['Pclass', 'Sex', 'Age','Survived']]
num_cols.isna().sum()


# In[41]:



from sklearn.impute import SimpleImputer
Imputer =SimpleImputer(missing_values=np.nan,strategy = 'constant')
Imputer =Imputer.fit(num_cols)
num_cols =Imputer.transform(num_cols)
type(num_cols)


# In[ ]:





# In[27]:


num_cols = pd.DataFrame(num_cols,columns = ['Pclass', 'Sex', 'Age','Survived'])
type(num_cols)


# In[28]:


num_cols.isna().sum()


# In[29]:


df = pd.concat([num_cols,df],axis=1)
df


# In[30]:


df.isna().sum()


# In[33]:


df1 = pd.read_csv("C:\\Users\\Noel\\Downloads\\titanic_dataset.csv", index_col = 'PassengerId')
freq_graph = df.select_dtypes(include='float')
freq_graph.hist(figsize = [20,15]);
                              


# In[40]:


df1['Age']=df1['Age'].fillna(df1['Age'].median())
df1['Age'].isna().sum()


# In[42]:


df1['Fare']=df1['Fare'].fillna(df1['Fare'].median())
df1['Fare'].isna().sum()


# In[41]:


df1.columns


# In[25]:



for i in['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
       'Fare', 'Cabin', 'Embarked']:
    df1[i] = df1[i].fillna(df1[i])
df1.isna().sum()    


# In[43]:


# Position of the Outlier
print(np.where(df.select_dtypes(include='float')>10))


# In[11]:


#Detecting the outliers
import seaborn as sns
sns.boxplot(df1['Fare'])


# In[12]:


df1 = pd.read_csv("C:\\Users\\Noel\\Downloads\\titanic_dataset.csv", index_col = 'PassengerId')
Q1=np.percentile(df1['Fare'],25)
Q1


# In[13]:


Q2=np.percentile(df1['Fare'],50)
Q2


# In[14]:


Q3=np.percentile(df1['Fare'],75)
Q3


# In[15]:


IQR=Q3-Q1
IQR


# In[17]:


low_limit=Q1-1.5*IQR
upr_lim=Q3+1.5*IQR
low_limit


# In[18]:


upr_lim


# In[29]:


outlier = []
for x in df1['Fare']:
    if((x>upr_lim)or(x<low_limit)):
        outlier.append(x)
outlier
    


# In[30]:


ind = df1['Fare']>upr_lim
df.loc[ind].index


# In[32]:


df1.drop([ 2,  28,  32,  35,  53,  62,  63,  73,  89, 103,
            
            793, 803, 821, 830, 836, 847, 850, 857, 864, 880],inplace=True)
df1.shape


# In[35]:


sns.boxplot(df1['Fare'])


# In[47]:


outlier = []
for x in df1['Age']:
    if((x>upr_lim)or(x<low_limit)):
        outlier.append(x)
outlier


# In[ ]:





# In[37]:


sns.boxplot(df1['Age'])


# In[ ]:





# In[8]:


#min_max scaling
df=pd.read_csv("C:\\Users\\Noel\\Downloads\\titanic_dataset.csv", index_col ='PassengerId')
x=df.drop('Parch',axis=1)
y=df['Parch']
x


# In[62]:


y


# In[79]:


x=x.drop('Embarked',axis=1)
x


# In[9]:



from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x['Survived'] = label_encoder.fit_transform(x['Survived'])
x


# In[10]:


x1=x.drop('Survived',axis=1)
x1


# In[27]:



from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler().fit(df[['Fare']])
df_copy = df.copy(deep=True)
df_copy['Fare_minmax'] =min_max.transform(df_copy[['Fare']])
print(df_copy.head())


# In[ ]:




