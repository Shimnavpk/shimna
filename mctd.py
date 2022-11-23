#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("C:\\Users\\Noel\\Downloads\\StudentsPerformance.csv")
data.head()


# In[16]:


#No.of males and females participated in the test.

result = data.pivot_table(columns=['gender'], aggfunc='count')
result


# In[10]:


#students' parental level of education

data['parental level of education'].unique()


# In[13]:


#average for math, reading and writing based on'gender','test preparation course'

data1 = data.groupby(['gender','test preparation course'])[['reading score', 'math score', 'writing score']].mean()
data1


# In[17]:


#scoring variation for math, reading and writing score

print(np.var(data))


# In[14]:


data['math score'].describe(percentiles=[0.25])


# In[ ]:





# In[ ]:




