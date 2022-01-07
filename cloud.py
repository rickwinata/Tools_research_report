#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot

funding = pd.read_csv("funding.csv")

funding.head()


# In[10]:


totalfunding = pd.read_csv("totalfunding.csv")

totalfunding.head()


# In[14]:


company = pd.read_csv("company.csv")

company.head()


# In[7]:


ax=sns.barplot( y="vertical", x="percent",data=funding,order=funding.sort_values('percent',ascending = False).vertical)
for i in ax.containers:
    ax.bar_label(i,)


ax.set_xlabel('Funding percentage',fontsize = 12) 
ax.set_ylabel('Verticals', fontsize = 12)


# In[11]:


plt.figure(figsize = (15, 10))

ax=sns.barplot( y="tf", x="time",data=totalfunding)
for i in ax.containers:
    ax.bar_label(i,)


ax.set_xlabel('Time',fontsize = 12) 
ax.set_ylabel('Total funding (Mil USD)', fontsize = 12)


# In[16]:


ax=sns.barplot( y="company", x="f",data=company,order=company.sort_values('f',ascending = False).company)
for i in ax.containers:
    ax.bar_label(i,)


ax.set_xlabel('Funding amount (Mil USD)',fontsize = 12) 
ax.set_ylabel('Startups', fontsize = 12)


# In[ ]:




