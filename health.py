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

heart = pd.read_csv("heart.csv")

heart.head()


# In[2]:


diabetes = pd.read_csv("diabetes.csv")

diabetes.head()


# In[8]:


ax=sns.lmplot(data=heart, x="age", y="chol",line_kws={'color': 'orange'})
ax = plt.gca()

ax.set_title('Age to Cholestrol Level',fontsize = 15)
ax.set_xlabel('Age',fontsize = 12) 
ax.set_ylabel('Cholestrol (mg/dl)', fontsize = 12)


plt.show()


# In[14]:


ax=sns.lmplot(data=diabetes, x="Age", y="DiabetesPedigreeFunction",line_kws={'color': 'orange'})
ax = plt.gca()

ax.set_title('Age to Diabetes Pedigree',fontsize = 15)
ax.set_xlabel('Age',fontsize = 12) 
ax.set_ylabel('Diabetes Pedigree', fontsize = 12)


plt.show()


# In[30]:


obesity = pd.read_csv("obesity2.csv")

obesity['Percentage'] = pd.to_numeric(obesity['Percentage'],errors = 'coerce')

obesity.dtypes


# In[32]:


ax=sns.lmplot(data=obesity, x="Year", y="Percentage",hue='Sex', order=2, ci=None, scatter_kws={"s": 80})
ax = plt.gca()

ax.set_title('Obesity Rate',fontsize = 15)
ax.set_xlabel('Year',fontsize = 12) 
ax.set_ylabel('Percentage', fontsize = 12)


plt.show()


# In[35]:


accident = pd.read_csv("vehicle.csv")

accident.head()


# In[68]:


motor = ['Cyclist','Motorcycle 125cc and under rider or passenger',
'Motorcycle 50cc and under rider or passenger',
'Motorcycle over 125cc and up to 500cc rider or  passenger'
]

accident2=accident.loc[accident['vehicle'].isin(motor)]
plt.figure(figsize = (20, 7))

ax=sns.countplot(data=accident2, x="vehicle",order=motor)
ax = plt.gca()

ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=20)
ax.set_title('Accident by Vehicle Type',fontsize = 15)
ax.set_xlabel('Vehicle',fontsize = 12) 
ax.set_ylabel('Count', fontsize = 12)


plt.show()


# In[51]:


accident2['date3'] = pd.to_datetime(accident2['date3'], errors = 'coerce')

accident2.dtypes


# In[70]:


accident2['year']=pd.DatetimeIndex(accident2['date3']).year
accident2.head()


# In[86]:


plt.figure(figsize = (15, 10))
ax=sns.countplot(data=accident2,x='year',hue='vehicle')
ax.set_title('Accident by Vehicle Type',fontsize = 15)
ax.set_xlabel('Year',fontsize = 12) 
ax.set_ylabel('Count', fontsize = 12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:




