#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np

import pandas as pd

import seaborn as sns
import scipy.stats as stats

import matplotlib.pyplot as plt
df=pd.read_excel("C:\\Users\\Noel\\Downloads\\iris (1).xls")
df.head()


# In[20]:


df.info()


# In[21]:


df.describe()


# In[22]:


df.isna().sum()


# In[36]:


df.dropna(axis = 1)


# In[44]:


df1 = df.fillna(df.mean())
df1


# In[38]:


df.notnull()


# In[45]:


df.isna().sum()


# In[23]:


df.columns


# In[24]:


df.dtypes


# In[25]:


df.shape


# In[26]:


df.size


# In[27]:


df.count()


# In[28]:


df['Classification'].value_counts()


# In[22]:


df['Classification'].unique()


# In[30]:


corr_matrix=df.corr()
sns.heatmap(corr_matrix ,annot=True)
plt.plot()


# In[40]:


df1 = pd.get_dummies(df)
df1


# In[76]:


x.shape


# In[41]:



x=df1.drop('SL',axis=1)
y=df1['PW']


# In[57]:


from sklearn import utils
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x['SW'] = label_encoder.fit_transform(x['SW'])
x


# In[78]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state =101,test_size=0.33)


# In[80]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model = log_model.fit(x_train,y_train)
y_pred = log_model.predict(x_test)


# In[82]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score,precision_score
print('accuracy is=:',accuracy_score(y_test,y_pred))


# In[83]:


y_pred


# In[84]:


y_test


# In[85]:


confusion_matrix(y_test,y_pred)


# # KNN

# In[87]:


from sklearn.neighbors import KNeighborsClassifier


# In[88]:


metric_k = []
neighbors = np.arange(3,15)


# In[89]:


for k in neighbors:
    classifier = KNeighborsClassifier(n_neighbors = k,metric = 'euclidean')
    classifier.fit(x_train,y_train)
y_prediction = classifier.predict(x_test)
acc = accuracy_score(y_test,y_prediction)
metric_k.append(acc)
metric_k


# In[90]:


x.shape 


# In[94]:


classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'euclidean')
classifier.fit(x_train,y_train)
y_prediction = classifier.predict(x_test)
print('accuracy is=:',accuracy_score(y_test,y_prediction))


# In[95]:


confusion_matrix(y_test,y_prediction)


# # SVM-linear

# In[99]:


from sklearn.svm import SVC
svm_cls = SVC(kernel='linear')
svm_cls = svm_cls.fit(x_train,y_train)
y_pred_svm = svm_cls.predict(x_test)


# In[100]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[101]:


confusion_matrix(y_test,y_pred_svm)


# In[102]:


accuracy_score(y_test,y_pred_svm)


# # SVM-RBF

# In[103]:


svm_cls1 = SVC(kernel='rbf')
svm_cls1 = svm_cls1.fit(x_train,y_train)
y_pred_svm1 = svm_cls1.predict(x_test)


# In[104]:


confusion_matrix(y_test,y_pred_svm1)


# In[105]:


accuracy_score(y_test,y_pred_svm1)


# # Decisiontree

# In[106]:


from sklearn.tree import DecisionTreeClassifier
dt_cls = DecisionTreeClassifier()
dt_cls = dt_cls.fit(x_train,y_train)
y_pred_dt= dt_cls.predict(x_test)


# In[107]:


confusion_matrix(y_test,y_pred_dt)


# In[108]:


accuracy_score(y_test,y_pred_dt)


# # RandomForest

# In[109]:


from sklearn.ensemble import RandomForestClassifier
rf_cls = RandomForestClassifier()
rf_cls =rf_cls.fit(x_train,y_train)
y_pred_rf= rf_cls.predict(x_test)


# In[110]:


confusion_matrix(y_test,y_pred_rf)


# In[111]:


accuracy_score(y_test,y_pred_rf)


# In[ ]:




