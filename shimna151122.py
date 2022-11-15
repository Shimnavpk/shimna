#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd

data= {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

df=pd.DataFrame(data,index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
print(df)


# In[ ]:


Dataframe is created with animals name,age,visits and priority.


# In[6]:


df1=df.head(3)
df1


# In[11]:


df[['animal','age']]


# In[15]:


df.loc[['d','e','i'],['animal', 'age']]


# 

# In[44]:



df['visits']>3


# In[45]:



df['visits']<3


# In[50]:


import numpy as np
import pandas as pd

data= {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

df=pd.DataFrame(data,index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
df['age']>2 and df['age']<4


# In[ ]:





# In[ ]:





# In[ ]:




