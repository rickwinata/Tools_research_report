#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv("indexdelivery.csv")

df.head()


# In[14]:


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot

fig,axes = plt.subplots(nrows=2,ncols=2,dpi=120,figsize = (8,6))

plot00=sns.distplot(df['Age'],ax=axes[0][0],color='red')
axes[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axes[0][0].set_title('Distribution of Age',fontdict={'fontsize':8})
axes[0][0].set_xlabel('Age',fontdict={'fontsize':7})
axes[0][0].set_ylabel('Count/Dist.',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.distplot(df['Family size'],ax=axes[0][1],color='blue')
axes[0][1].set_title('Distribution of Family Size',fontdict={'fontsize':8})
axes[0][1].set_xlabel('Family Size',fontdict={'fontsize':7})
axes[0][1].set_ylabel('Count/Dist.',fontdict={'fontsize':7})
axes[0][1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.tight_layout()


plot10=sns.boxplot(df['Age'],ax=axes[1][0],color='red')
axes[1][0].set_title('Age Distribution',fontdict={'fontsize':8})
axes[1][0].set_xlabel('Distribution',fontdict={'fontsize':7})
axes[1][0].set_ylabel(r'Age',fontdict={'fontsize':7})
plt.tight_layout()


plot11=sns.boxplot(df['Family size'],ax=axes[1][1],color='blue')
axes[1][1].set_title(r'Numerical Summary (Family Size)',fontdict={'fontsize':8})
axes[1][1].set_ylabel(r'Size of Family',fontdict={'fontsize':7})
axes[1][1].set_xlabel('Family Size',fontdict={'fontsize':7})
plt.tight_layout()

plt.show()


# In[31]:



plt.figure(figsize = (15, 7))



plt.subplot(2,3,1)
ax = sns.countplot(x="Gender", data=df,
                  linewidth = 3,
                   edgecolor='dimgray')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
ax.set_title('Gender',fontsize = 15)
ax.set_xlabel('Types',fontsize = 12) 
ax.set_ylabel('Count', fontsize = 12)
plt.tight_layout()


plt.subplot(2,3,2)
ax = sns.countplot(x="Marital Status", data=df,
                  linewidth = 3,
                   edgecolor='dimgray')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
ax.set_title('Marital Status',fontsize = 15)
ax.set_xlabel('Types',fontsize = 12) 
ax.set_ylabel('Count', fontsize = 12)
plt.tight_layout()


plt.subplot(2,3,3)
ax = sns.countplot(x="Occupation", data=df,
                  linewidth = 3,
                   edgecolor='dimgray')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
ax.set_title('Occupation',fontsize = 15)
ax.set_xlabel( 'Types',fontsize = 12) 
ax.set_ylabel('Count', fontsize = 12)
plt.tight_layout()


plt.subplot(2,3,4)
ax = sns.countplot(x="Age", data=df,
                  linewidth = 3,
                   edgecolor='dimgray')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
ax.set_title('Age',fontsize = 15)
ax.set_xlabel('Types',fontsize = 12) 
ax.set_ylabel('Count', fontsize = 12)
plt.tight_layout()


plt.subplot(2,3,5)
ax = sns.countplot(x="Monthly Income", data=df,
                  linewidth = 3,
                   edgecolor='dimgray')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=20)
ax.set_title('Monthly Income',fontsize = 15)
ax.set_xlabel('Types',fontsize = 12) 
ax.set_ylabel('Count', fontsize = 12)
plt.tight_layout()


plt.subplot(2,3,6)
ax = sns.countplot(x="Family size", data=df,
                  linewidth = 3,
                   edgecolor='dimgray')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=20)
ax.set_title('Family size',fontsize = 15)
ax.set_xlabel('Types',fontsize = 12) 
ax.set_ylabel('Count', fontsize = 12)
plt.tight_layout()


# In[15]:



plt.figure(figsize = (15, 7))
plt.style.use('bmh')
order_list = ['15 minutes', '30 minutes', '45 minutes', '60 minutes', 'More than 60 minutes']
order_list2 = ['Weekdays (Mon-Fri)','Weekend (Sat & Sun)', 'Anytime (Mon-Sun)']


plt.subplot(1,2,1)
ax = sns.countplot(x="Order Time", data=df,
                  linewidth = 3,
                   edgecolor='dimgray',order=order_list2)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
ax.set_title('Order Time',fontsize = 15)
ax.set_xlabel('Types',fontsize = 12) 
ax.set_ylabel('Count', fontsize = 12)
plt.tight_layout()


plt.subplot(1,2,2)
ax = sns.countplot(x="Maximum wait time", data=df,
                  linewidth = 3,
                   edgecolor='dimgray',order=order_list)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
ax.set_title('Maximum wait time',fontsize = 15)
ax.set_xlabel( 'Types',fontsize = 12) 
ax.set_ylabel('Count', fontsize = 12)
plt.tight_layout()


# In[65]:


plt.figure(figsize = (18, 7))

order_list = ['15 minutes', '30 minutes', '45 minutes', '60 minutes', 'More than 60 minutes']
order_list2 = ['Weekdays (Mon-Fri)','Weekend (Sat & Sun)', 'Anytime (Mon-Sun)']

plt.subplot(1,2,1)
ax=sns.boxplot(data=df, x='Order Time', y='Age',order=order_list2)
ax.set_title('Age to Order time',fontsize = 15)
plt.tight_layout()

plt.subplot(1,2,2)
ax=sns.boxplot(data=df,x='Maximum wait time', y='Age',order=order_list)
ax.set_title('Age to Waiting time',fontsize = 15)
plt.tight_layout()


# In[24]:


import seaborn as sns


lm= sns.lmplot(data=df, x="INDEX_PREF", y="INDEX_RATING", hue="Gender")

ax = plt.gca()

ax.set_title('Preference to Rating',fontsize = 15)
ax.set_xlabel('Preference Index',fontsize = 12) 
ax.set_ylabel('Rating Index', fontsize = 12)


plt.show()


# In[31]:


import geopandas as gpd
import math
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster

Age_band = df[(df.Age.isin(range(18,40)))]

m_2 = folium.Map(location=[12.9716,77.5946], zoom_start=13,)


for idx, row in Age_band.iterrows():
    Marker([row['latitude'], row['longitude']]).add_to(m_2)


m_2


# In[32]:


m_3 = folium.Map(location=[12.9716,77.5946], zoom_start=13)


mc = MarkerCluster()
for idx, row in Age_band.iterrows():
    if not math.isnan(row['longitude']) and not math.isnan(row['latitude']):
        mc.add_child(Marker([row['latitude'], row['longitude']]))
m_3.add_child(mc)

m_3


# In[57]:


df2 = pd.read_csv("foodtype2.csv")
df2.head()

ax=sns.lmplot(data=df2, x="votes", y="calories",line_kws={'color': 'orange'})
ax = plt.gca()

ax.set_title('Order to Calories',fontsize = 15)
ax.set_xlabel('Order',fontsize = 12) 
ax.set_ylabel('Calories per 100g', fontsize = 12)


plt.show()


# In[51]:


df2.head()


# In[58]:


ax=sns.lmplot(data=df2, x="calories", y="rate",line_kws={'color': 'orange'},x_estimator=np.mean)
ax = plt.gca()

ax.set_title('Calories to Rating',fontsize = 15)
ax.set_xlabel('Calories per 100g',fontsize = 12) 
ax.set_ylabel('Rating', fontsize = 12)


plt.show()


# In[98]:


cal = pd.read_csv("calories.csv")

cal.head()


# In[99]:


plt.figure(figsize = (15, 20))
ax=sns.barplot( y="cuisine", x="calories",data=cal,order=cal.sort_values('calories').cuisine)
for i in ax.containers:
    ax.bar_label(i,)

ax.set_title('Calories Based on Cuisine',fontsize = 15)
ax.set_xlabel('Calories per 100g',fontsize = 12) 
ax.set_ylabel('Cuisines', fontsize = 12)


# In[ ]:




