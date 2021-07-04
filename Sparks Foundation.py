#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation Internship
# ### By: Deepansh Bhatnagar
# 
# ## Task 1- Prediction Using Supervised ML
# Linear Regression
# 
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied

# In[1]:

#aur bhai
#importing library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[26]:


#importing data
data=pd.read_csv("http://bit.ly/w-data")
data


# In[ ]:





# In[3]:


#plotting
plt.plot(data["Hours"],data["Scores"],'r o')
#labeling
plt.title("Hours vs Percentage")
plt.xlabel("Hours")
plt.ylabel("Percentage")
plt.grid(True)


# In[27]:


#spliting the data into 2 parts 
inp=data.iloc[:,:-1].values
out=data.iloc[:,-1].values


# In[39]:


#test and train split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inp,out,test_size=0.2,random_state=12)


# In[42]:


x_train.shape,x_test.shape


# In[38]:


#SINCE LINEAR REGRESSION 
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
print("TRAINING DONE!!!!!!!!")


# In[30]:


#INTERCEPT
inter=model.intercept_


# In[31]:


#coefficent
coeff=model.coef_


# In[32]:


#for regression line
Rline=coeff*inp+inter
#ploting the test data
plt.scatter(inp,out)
plt.plot(inp,Rline,'r--')
plt.grid(True)
plt.tight_layout()


# # Now Predictions

# In[33]:


print(x_test)
ypred=model.predict(x_test)
ypred


# In[34]:


#comparing actual vs predicted
df=pd.DataFrame({"Actual":y_test,"Predicted":ypred})
df


# In[35]:


#evaluation of the model
from sklearn import metrics
print("Mean Absolute Error :",metrics.mean_absolute_error(y_test,ypred))
print("Mean Squared Error :",metrics.mean_squared_error(y_test,ypred))
print("Mean Absolute Error :",np.sqrt(metrics.mean_squared_error(y_test,ypred)))


# In[47]:


new=model.predict([[9.5]])
new


# In[ ]:




