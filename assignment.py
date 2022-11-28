#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("C:\\Users\\Noel\\Downloads\\Sales_add (2).csv")
data


# In[7]:


#H0-Not increase in sales after stepping into digital marketing.
#H1-Increase in sales after stepping into digital marketing.
import statistics as stat
stat.variance(data['Sales_After_digital_add(in $)'])


# In[8]:


stat.stdev(data['Sales_After_digital_add(in $)'])


# In[42]:


stat.mean(data['Sales_before_digital_add(in $)'])


# In[44]:


stat.mean(data['Sales_After_digital_add(in $)'])


# In[51]:


from statsmodels.stats.weightstats import ttest_ind
data=pd.read_csv("C:\\Users\\Noel\\Downloads\\Sales_add (2).csv")
ttest_ind(data['Sales_before_digital_add(in $)'], data['Sales_After_digital_add(in $)'])


# In[53]:


from scipy.stats import ttest_ind
t_stat1,p_value1=ttest_ind(data['Sales_before_digital_add(in $)'], data['Sales_After_digital_add(in $)'])

p_value1


# In[54]:


if p_value1<0.05:
     print("Null hypothesis rejected")
else:
      print("Null hypothesis accepted")


# In[ ]:


#ie,Increase in sales after stepping into digital marketing.


# In[6]:


import scipy

corr, p_values = scipy.stats.spearmanr(data['Region'],data['Manager'])
print(corr, p_values)
    
    


# In[ ]:


#It means that there is a strong positive correlation between 'Region'and'Manager'


# In[ ]:




