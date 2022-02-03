#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import os
import statsmodels.api as sm


# In[37]:


os.chdir = (R'C:\Users\Pratik G Ratnaparkhi\Desktop\IVY Python\Python_7')
path_data = os.getcwd()
weather = pd.read_csv('weatherHistory.csv')


# In[38]:


display(weather.head(3))


# In[39]:


#Changing Col Names


# In[40]:


weather = pd.DataFrame(weather)
weather = weather.rename(columns = {"Precip Type":"PrecipType","Temperature (C)":"Temperature",
                        "Apparent Temperature (C)":"AppTemp",
                       "Wind Speed (km/h)":"WindSpeed",
                       "Wind Bearing (degrees)":"WindBearing",
                       "Visibility (km)":"Visibility",
                       "Pressure (millibars)":"Pressure"})
weather.head()


# In[41]:


weather.isnull().sum()


# In[42]:


weather.PrecipType.unique()


# In[43]:


plt.subplot(1,2,1)
sns.boxplot(data=weather,y='Temperature',x='PrecipType')
plt.subplot(1,2,2)
sns.boxplot(data=weather,y='Humidity',x='PrecipType')
plt.tight_layout()


# In[44]:


weather['PrecipType'].fillna("snow",inplace=True)


# In[45]:


weather['PrecipType'].value_counts().plot(kind='bar')


# In[46]:


#Lets check null values again
weather.isnull().sum()


# In[47]:


weather.describe(include='all')


# In[48]:


a = weather["Temperature"]
b = weather["Humidity"]


# In[49]:


sns.regplot(x=a,y=b,color="blue",marker='*')


# In[50]:


display(weather.head())


# In[51]:


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.boxplot(x=weather['Humidity'])

plt.subplot(2,2,2)
sns.boxplot(x=weather['AppTemp'])


# In[52]:


#removing outliers from humidity by IQR method
Q1 = weather.Humidity.quantile(0.25)
Q3 = weather.Humidity.quantile(0.75)
IQR = Q3-Q1
print("{},{},{}".format(Q1,Q3,IQR))


# In[53]:


lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
print(upper)
print(lower)


# In[54]:


weather_1 = weather[(weather.Humidity>lower)&(weather.Humidity<upper)]
weather_1.head()


# In[55]:


#Lets check outliers in humidity again
sns.boxplot(x=weather_1['Humidity'])


# In[56]:


#Building MLR moddel


# In[57]:


#creating dependent and independent variable
y= weather.iloc[:,[3]]
y


# In[58]:


x= weather.iloc[:,[2,4,5]]
x


# In[62]:


#creating dummies for col "Precip Type" as it is having categorical variable
x2 = pd.get_dummies(x, columns=['PrecipType'])
x2


# In[63]:


#checking Multicollinearity
x2 = sm.add_constant(x2)


# In[64]:


model = sm.OLS(y,x2).fit()
model.summary()


# In[ ]:


#data is have R2 = 0.98 which is good 
#Also no signs of Multicollinearity in data as values of std error are less


# In[65]:


from sklearn.model_selection import train_test_split


# In[66]:


x_train,x_test,y_train,y_test = train_test_split(x2,y,test_size=0.2,random_state=0)


# In[67]:


from sklearn.linear_model import LinearRegression


# In[68]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[69]:


print(regressor.coef_)
print(regressor.intercept_)


# In[79]:


y_pred = regressor.predict(x_test)
y_pred


# In[80]:


display(regressor.score(x_train,y_train))
display(regressor.score(x_test,y_test))


# In[82]:


from sklearn.metrics import mean_absolute_error
mape=mean_absolute_error(y_pred,y_test)
print("Mean Absolute Error is :" , mape)


# In[72]:


#Conclusion
"""We have achieved regressor score  around 98% for both train and test 
   data """
"""From MAE we can say that our predicted values are
   less than or greater than 0.82 units"""


# In[ ]:





# In[ ]:




