#!/usr/bin/env python
# coding: utf-8

# In[30]:


#Logistic regression Case Study
import numpy as np
import pandas as pd 
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix


# In[3]:


os.chdir(R'C:\Users\Pratik G Ratnaparkhi\Desktop\IVY Python\Python_8log')
path_data = os.getcwd()
data = pd.read_csv('Social_Network_Ads.csv')


# In[4]:


data.head()


# In[4]:


#Here Purchase is dependent variable


# In[5]:


#here we will do EDA(Exploratory Data Analysis)
x = data["Age"]
y = data["EstimatedSalary"]
plt.scatter(x,y)
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Estimated Salay')
plt.show()


# In[6]:


x = data["Age"]
y1 = data["Purchased"]
plt.scatter(x,y1)
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Purchased or not')
plt.show()


# In[7]:


x1 = data["Purchased"]
sns.countplot(x=x1,data=data,palette='hls')
plt.show()


# In[35]:


sns.heatmap(data.corr())


# In[8]:


#Check the missing value 
data.isnull().sum()


# In[9]:


data['EstimatedSalary'].describe()


# In[10]:


data['Gender'].unique()


# In[11]:


#Replacing Male Female to 0 1 


# In[12]:


data=pd.DataFrame(data)
data['Gender'].replace({'Female':0,'Male':1},inplace=True)
data


# In[13]:


# creating Dependent and Independent Variable
x2 = data.iloc[:,[1,2,3]].values
x2


# In[14]:


y2 = data.iloc[:,4].values
y2


# In[15]:


x2 = pd.DataFrame(x2)


# In[16]:


"""Basic Steps involved in Logistic Regression
1. Spliting data into training and testing
2. Feature Scling(if required)
3. Fitting
4. Predicting """


# In[17]:



x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size = 0.25, random_state = 0) 


# In[18]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[20]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[22]:


y_pred = classifier.predict(x_test)
y_pred


# In[28]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[26]:


import statsmodels.api as sm
log = sm.Logit(y_train,x_train)
result = log.fit(method='bfgs')
print(result.summary())


# In[32]:


#Checking model acurracy using classification_report
print(classification_report(y_test,y_pred))


# In[34]:


#Another method to check acurracy
classifier.score(x_test,y_test)#Acurracy Score 


# In[ ]:


""""From this we can conclude that this model is 90% times
    predicting acurrately weather user purchased or not"""
    


# In[ ]:




