#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[68]:


#  Create New Variable and stores the dataset values as Data Frame

train=pd.read_csv("C:\\Users\\Noel\\Downloads\\train_ctrUa4K.csv")
test=pd.read_csv("C:\\Users\\Noel\\Downloads\\test_lAUu6dG.csv") 
submission = pd.read_csv("C:\\Users\\Noel\\Downloads\\sample_submission_49d68Cx.csv") 


# In[69]:


train.head()


# In[70]:


test.head()


# In[71]:


submission.head()


# In[72]:


#shape

train.shape


# In[73]:


test.shape


# In[74]:


#Describe

train.describe()


# In[75]:


test.describe()


# In[76]:


train.info()


# In[77]:


test.info()


# In[78]:


submission.info()


# In[79]:


train['Loan_Status'].value_counts()


# In[81]:


train['Loan_Status'].value_counts(normalize=True) 


# In[82]:


#Visualisation of data
train.plot(figsize=(18, 8))
plt.show()


# In[83]:


Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()


# In[84]:


Married=pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()
Education.div(Education.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()


# In[85]:


train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# In[86]:


plt.figure(figsize=(18, 6))
plt.title("Relation Between Application Income vs Loan Amount ")

plt.grid()
plt.scatter(train['ApplicantIncome'] ,train['LoanAmount'], c='k', marker='x')
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.show()


# In[87]:


plt.figure(figsize=(12, 6))
plt.plot(train['Loan_Status'], train['LoanAmount'])
plt.title("Loan Application Amount ")
plt.show()


# In[88]:


train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'], axis=1)
train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+', 3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)


# In[89]:


matrix = train.corr()
f, ax = plt.subplots(figsize=(9,6))
sns.heatmap(matrix,vmax=.8,square=True,cmap='BuPu', annot = True)


# In[90]:


#checking null value
train.isna().sum()


# In[91]:



train.dropna(inplace=True)


# In[92]:


train.isna().sum()


# In[93]:


test.isna().sum()


# In[94]:


test.dropna(inplace=True)


# In[95]:


test.isna().sum()


# In[96]:


submission.isna().sum()


# In[97]:


submission.dropna(inplace=True)


# In[98]:


submission.isna().sum()


# In[99]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# In[100]:


train['Loan_Amount_Term'].value_counts()


# In[101]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


# In[102]:


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Married'].fillna(train['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


# In[103]:


train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)


# In[104]:


X = train.drop('Loan_Status',1)
y = train.Loan_Status


# In[105]:


X = pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)


# In[106]:



x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


# In[107]:



model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression()


# In[108]:


pred_test= model.predict(x_test)
accuracy_score(y_test,pred_test)


# In[ ]:


#Training the model


# In[112]:


#svm linear


svm_cls = SVC(kernel='linear')
svm_cls = svm_cls.fit(x_train,y_train)
y_pred_svm = svm_cls.predict(x_test)


# In[113]:


confusion_matrix(y_test,y_pred_svm)


# In[114]:


accuracy_score(y_test,y_pred_svm)


# In[115]:


#Decisiontree


dt_cls = DecisionTreeClassifier()
dt_cls = dt_cls.fit(x_train,y_train)
y_pred_dt= dt_cls.predict(x_test)


# In[116]:


confusion_matrix(y_test,y_pred_dt)


# In[117]:


accuracy_score(y_test,y_pred_dt)


# In[118]:


#RandomForest

rf_cls = RandomForestClassifier()
rf_cls =rf_cls.fit(x_train,y_train)
y_pred_rf= rf_cls.predict(x_test)


# In[119]:


confusion_matrix(y_test,y_pred_rf)


# In[120]:


accuracy_score(y_test,y_pred_rf)


# In[132]:


train1 = pd.DataFrame(train)
submission1 = pd.DataFrame(submission)
submission1 = pd.concat([train1,submission1],axis=1)
submission1.head()


# In[133]:


submission = submission.fillna(0)


# In[134]:


submission1 = submission.apply(lambda col: pd.Series(col.unique()))
submission1


# In[135]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv("C:\\Users\\Noel\\Downloads\\sample_submission_49d68Cx.csv")


# In[ ]:




