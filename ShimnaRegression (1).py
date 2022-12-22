#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np

import pandas as pd

import seaborn as sns
import scipy.stats as stats

import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\Noel\\Downloads\\car_age_price (2).csv")
df


# In[76]:


df.columns


# In[77]:


df['Year'].nunique()


# In[78]:


df['Price'].nunique()


# In[79]:


df.describe()


# In[80]:


df.info()


# In[6]:


df.isna().sum()


# In[54]:


df['Price'].mean()


# In[10]:


corr_matrix=df.corr()
corr_matrix
sns.heatmap(corr_matrix ,annot=True,cmap = 'Wistia')
plt.plot()


# In[15]:


df['Year'].unique()


# In[12]:


df1 = pd.get_dummies(df)
df1


# In[55]:


corr_matrix=df1.corr()
corr_matrix
sns.heatmap(corr_matrix ,annot=True)
plt.plot()


# In[40]:



x=df['Year'].values.reshape(-1,1)
y=df['Price'].values.reshape(-1,1)
x.shape


# In[35]:


# splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state =42,test_size=0.2)
x_train


# In[41]:


x_test


# In[59]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")



# In[60]:


print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")


# In[63]:


new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept: {new_model.intercept_}")
print(f"slope: {new_model.coef_}")


# In[64]:


y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")


# In[ ]:


#Value of(r_sq =0.50 )suggests that 50% of the dependent variable is predicted by the independent variableIt has low correlation


# In[74]:


from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train, y_train) 
pred_train_lasso= model_lasso.predict(X_train)
print('MSE=:',np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print('r2_score',r2_score(y_train, pred_train_lasso))

pred_test_lasso= model_lasso.predict(X_test)
print('MSE=:',np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print('r2_score',r2_score(y_test, pred_test_lasso))


# In[ ]:




