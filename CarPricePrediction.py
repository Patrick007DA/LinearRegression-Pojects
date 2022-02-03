#!/usr/bin/env python
# coding: utf-8

# # Cars Price Prediction using Linear    Regression

# In[181]:


import matplotlib.pyplot as plt
import pandas as pd
import os
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[42]:


car_price = pd.read_csv(R'C:\Users\Pratik G Ratnaparkhi\Desktop\IVY Python\CarPrices.csv')


# In[43]:


car_price.head()


# In[44]:


car_price.shape


# In[45]:


#Descriptive Analysis


# In[46]:


car_price.describe()


# In[47]:


car_price.info()


# In[48]:


#finding total null values


# In[49]:


car_price.isnull().sum()


# In[50]:


#Creating Dataframe


# In[51]:


car_price1 = pd.DataFrame(car_price)


# In[52]:


car_price1.head()


# In[53]:


cat = car_price1.iloc[:,[3,4,8,14,17,25]]


# In[54]:


cat.head()


# In[55]:


#Exploratory Data Analysis
#Finding Some relation between Categorical variables using pairplot from seaborn


# In[56]:


plt.figure(figsize=(20,15))
x=1
for i in cat:
    plt.subplot(3,3,x)
    sns.boxplot(data=cat,y='price',x=i)
    plt.ylabel('Price',fontsize=14)
    plt.xlabel(i,fontsize=14)
    plt.tight_layout()
    x=x+1


# In[57]:


num = car_price.iloc[:,[20,21,24,25]]
num.head()


# In[58]:


plt.figure(figsize=(8,8))
sns.pairplot(num)
plt.show()


# In[59]:


#Finding Correlation and Correaltion matrix 


# In[60]:


car_price1.corr()


# In[61]:


num1 = car_price1.iloc[:,[10,11,12,13,18,19,20,21,22,23,24,25]]


# In[62]:


df = num1.corr()
df


# In[63]:


hm = sns.heatmap(df)
hm


# In[64]:


#Price distribution by histogram


# In[65]:


plt.hist(car_price1['price'],bins=10,color='y')
plt.show()


# In[66]:


sns.distplot(car_price1['horsepower'])


# In[67]:


#Encoding categorical data by pd.get_dummies 
#to get started with it we have to make list of Cat_variable


# In[102]:


cat_var = ['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']


# In[103]:


car_price2 = pd.get_dummies(car_price1, columns=cat_var, drop_first=True)
car_price2.head()


# In[104]:


price = car_price2.drop(['car_ID','CarName'],axis=1)
price


# In[105]:


price.info()


# In[ ]:


#Feature Scaling


# In[137]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform((price.drop('price',axis = 1)))


# In[138]:


#Creating Dependent and Independent variable


# In[139]:


x = x_scaled
Y = price["price"]
x= pd.DataFrame(data=x,columns = price.drop(columns=['price']).columns)


# In[140]:


x.corr()


# In[ ]:


#Testing Multicollinearity using Variance Inflation Factor
#we are using user def function to calculate the vif values for all columns


# In[165]:


vif_data=x
VIF=pd.Series([variance_inflation_factor(vif_data.values,i) for i in range(vif_data.shape[1])],index=vif_data.columns)
VIF


# In[ ]:


#Removing columns with high multicollinearity using user def func


# In[166]:


def removecol(data):
    vif=pd.Series([variance_inflation_factor(data.values,i)for i in range(data.shape[1])],index=data.columns)
    if vif.max()>5:
        data = data.drop(columns=[vif[vif==vif.max()].index[0]])
        return data
    else:
        return data


# In[167]:


for i in range(10):
    vif_data=removecol(vif_data)
vif_data.head()


# In[168]:


VIF=pd.Series([variance_inflation_factor(vif_data.values,i) for i in range(vif_data.shape[1])],index=vif_data.columns)
VIF


# In[ ]:


#there are some vif values which are again >5
#we will apply multicollinearity test again


# In[173]:


def removecol2(data):
    vif=pd.Series([variance_inflation_factor(data.values,i)for i in range(data.shape[1])],index=data.columns)
    if vif.max()>5:
        data = data.drop(columns=[vif[vif==vif.max()].index[0]])
        return data
    else:
        return data


# In[175]:


for i in range(10):
    vif_data=removecol(vif_data)
vif_data.head()


# In[176]:


VIF=pd.Series([variance_inflation_factor(vif_data.values,i) for i in range(vif_data.shape[1])],index=vif_data.columns)
VIF


# In[178]:


vif_data = x


# In[180]:


y = pd.DataFrame(y)
y


# In[ ]:


#Creating Training and Testing data


# In[182]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[183]:


from sklearn.linear_model import LinearRegression


# In[185]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[197]:


y_pred = regressor.predict(x_test)
y_pred


# In[198]:


print(regressor.coef_)

print(regressor.intercept_)


# In[188]:


display(regressor.score(x_train,y_train))
display(regressor.score(x_test,y_test))


# In[193]:


#Calculating MeanAPE


# In[204]:


from sklearn.metrics import mean_absolute_percentage_error
mape=mean_absolute_percentage_error(y_pred,y_test)
print("Mean Absolute Percentage Error is :" , mape)


# In[ ]:


#Conclusion 
 """Our Predicted value can be 0.15 units less than or greater
     than the actual value"""


# In[ ]:


"""We Have achived reg score 95 % for training and
   86% for testing"""

