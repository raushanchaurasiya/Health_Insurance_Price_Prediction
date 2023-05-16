#!/usr/bin/env python
# coding: utf-8

# # Predicting Health Insurance Price for an individual or family

# The majority of the countries finalize health insurance costs based on many factors such as age, number of people in families, etc. What should be the actual health insurance price for an individual or a family is an issue for many companies. Hence, one insurance company hired you as a data scientist to predict the health insurance cost for possible future customers. They have already collected samples required to perform all data analysis and machine learning tasks. Your task is to perform all data analysis steps and finally create a machine learning model which can predict the health insurance cost.

# In[244]:


#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import plotly.express as px


# In[245]:


import warnings
warnings.filterwarnings("ignore") # ignoring wranings


# In[246]:


#reading the dataset

Health_raw = pd.read_excel("Health_insurance_cost (1).xlsx")
Health_data = Health_raw.copy() #creating a copy of raw dataset


# ### Exploratory Data Analysis

# In[247]:


#reading the top 5 records
Health_data.head()


# In[248]:


#reading the last 5 records
Health_data.tail()


# * **There are missing values in the form of NaN**
# * **In our dataset age , gender , BMI , Children , smoking_status , location are Independent Variables.**
# * **And , health_insurance_price is dependent Variable which is dependent on Independent Variables**
# * **As , we can see that health_insurance_price is continous Dependent Variable , So this is Regression Problem**

# In[249]:


#exploring the number of observations and variables
Health_data.shape


# In[250]:


#getting ststistical summary of dataset
Health_data.describe(include = 'all')


# * Data is looking good as Min and Max are possible values , there may be outliers but data does't contain impossible/incorrect values.
# * Age have two unique values Male and Female
# * smoking_status also have two unique values with YES or NO.
# * In location , there are 4 locations

# In[251]:


# checking the null values present and Datatypes of Features.

Health_data.info()


# **Observations**
# * we have 3 features with categorical data
# * 4 features with numerical data
# * health_insurance_price is the Target Variable
# * we have 2 null values in Target Variable

# In[252]:


#getting the sum of null values present in every feature

Health_data.isna().sum()


# In[253]:


#checking for duplicates and dropping duplicates , if any

Health_data.drop_duplicates()


# * There are no duplicates in the dataset.

# **Univariate Analysis**

# In[254]:


fig = px.histogram(Health_data, x="age", nbins=50, width=800, height=400, # creating histogram for life expectancy with 20 bins
                  labels={"age": "Age Distribution"})
fig.show()


# * People applying for Health Insurance is highest in number with age group 19-20 

# In[255]:


plt.style.use('ggplot')
plt.figure(figsize = [5,6])
Health_data['smoking_status'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True, startangle=60, labeldistance=None)
plt.legend()


# * approx 80% customers are non-smokers who are applying for Health Insurance.
# * Number of non-smokers applying for Health Insurance are maximum.

# In[256]:


fig = px.histogram(Health_data, x="gender", width=500, height=400, # creating histogram for life expectancy with 20 bins
                  labels={"gender": "Gender Distribution"})
fig.show()


# In[257]:


fig = px.histogram(Health_data, x="BMI" ,width=800, height=400, # creating histogram for life expectancy with 20 bins
                  labels={"BMI": "BMI of Customers"})
fig.show()


# In[258]:


# binning 'BMI' and Creating new column "BMI_range"
Health_raw['BMI_range']= pd.cut(Health_raw['BMI'], [0,18.5,24.9,29.9,100],labels=['UnderWeight','Healthy','Overweight','Obese'])
(Health_raw['BMI_range'].value_counts(normalize=True)*100).plot.barh(title ="Body Mass Index Group of customers", color=['indianred', 'dodgerblue', 'darkcyan', 'lightslategrey', 'lightseagreen' ])
plt.xticks(rotation=0)
plt.show()


# * Normal BMI Range = 18.5 - 24.9
# * The Overweight and Obese Group are the largest group applying for Health Insurance.
# * This can affect the Insurance Price the person get.

# In[259]:


fig = px.histogram(Health_data, x="Children", width=800, height=400, # creating histogram for life expectancy with 20 bins
                  labels={"Children": "Number of Childrens"})
fig.show()


# * Customers Applying for Health Insurance who have NO CHILDREN are maximum.

# In[260]:


fig = px.histogram(Health_data, x="location", width=800, height=400, # creating histogram for life expectancy with 20 bins
                  labels={"location": "Location of Customers"})
fig.show()


# In[261]:


# distribution of charges value
plt.figure(figsize=(6,6))
sns.distplot(Health_data['health_insurance_price'])
plt.title('Insurance Price Distribution')
plt.show()


# * we have a lot of data distributed between 10000 and we have very little values in 30,000 -40,000.

# **Checking for Outliers**

# In[262]:


fig = px.box(Health_data, y="age", width=400, height=300)
fig.show()


# * Data is normally distributed. Min - 18 & Max - 64

# In[263]:


fig = px.box(Health_data, y="BMI", width=400, height=300)
fig.show()


# * There are outliers in BMI Data , But these are not incorrect data as these are possible so we will keep as it is.

# In[264]:


fig = px.box(Health_data, y="health_insurance_price", width=400, height=300)
fig.show()


# * There are outliers too , but again these are possible data.

# In[265]:


ax = sns.lmplot(x = 'age', y = 'health_insurance_price', data=Health_data, hue='smoking_status', palette='Set1')
ax = sns.lmplot(x = 'BMI', y = 'health_insurance_price', data=Health_data, hue='smoking_status', palette='Set2')
ax = sns.lmplot(x = 'Children', y = 'health_insurance_price', data=Health_data, hue='smoking_status', palette='Set3')


# **As we can see , Smoking has serious effect on Health_insurance_price.**

# In[266]:


#First we will do a correlation matrix heatmap to see the correlation coefficient for each pair of features

corr_matrix = Health_data.corr(method='spearman').abs()

plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, cmap="Reds")
plt.show()


# **We can see that Health Insurance Price are strong connected with age and if the person is smoker or not**

# In[267]:


Health_data.groupby("smoking_status").agg({"health_insurance_price": "mean"})


# **Observations**
# * We can see the changes in the amount of Health Insurance Price for Smokers and Non-smokers.
# * There is a huge difference between both of the groups.

# ### Dealing with Null Values

# We have null values in three features Age , BMI and health_insurance_price.
# 
# health_insurance_price is our Target Variable , so we can't impute null values.
# We have to remove it.

# In[268]:


#checking for any pattern in records where null values present in health_insurance_price

Health_data[Health_data['health_insurance_price'].isnull()]


# In[269]:


#dropping both records

Health_data.dropna(subset=['health_insurance_price'], inplace=True)


# In[270]:


Health_data.info()


# * we are now left with 1336 records.

# In[271]:


Health_data.isna().sum()


# In[272]:


# visualization of missing values
import missingno as msno
msno.matrix(Health_data)


# In[273]:


# msno.bar is a simple visualization of null values by column:

msno.bar(Health_data.sample(1000))


# In[274]:


# To check the missing no. correlation heatmap measures nullity correlation.

msno.heatmap(Health_data)


# ### Converting Categorial Columns into numerical Columns

# we have 3 features with String datatype , As ML Algoriths understand only numerical columns so we need to convert these features into numerical data.
# 

# In[275]:


#fetching uniques values of features
print(Health_data['gender'].unique())
print(Health_data['smoking_status'].unique())
print(Health_data['location'].unique())


# In[276]:


#Encoding Gender column
Health_data['gender'] = Health_data['gender'].map({'female' : 0 , 'male' : 1}) 
#Encoding Smoking_status Column
Health_data['smoking_status'] = Health_data['smoking_status'].map({'no' : 0 , 'yes' : 1}) 
#Encoding Location Column
Health_data['location'] = Health_data['location'].map({'southwest' : 1 , 'southeast' : 2 , 'northwest' : 3 , 'northeast' : 4})


# In[277]:


Health_data.head()


# ### Imputation of Null Values

# **Mean Imputation**

# In[278]:


from sklearn.impute import SimpleImputer # importing simple imputer
health_mean = Health_data.copy(deep=True) # creating a copy for further evaluation
mean_imputer = SimpleImputer(strategy='mean') # using mean to impute the value
health_mean.iloc[:, :] = mean_imputer.fit_transform(health_mean) #  both fit a model to the data and then transform the data using that model


# **Median Imputation**

# In[279]:


health_median = Health_data.copy(deep=True) # creating a copy for further evaluation
median_imputer = SimpleImputer(strategy='median') # using median to impute the value
health_median.iloc[:, :] = median_imputer.fit_transform(health_median)


# **Mode Imputation**

# In[280]:


health_mode = Health_data.copy(deep=True) # creating a copy for further evaluation
mode_imputer = SimpleImputer(strategy='most_frequent') # using most frequent to impute the value
health_mode.iloc[:, :] = mode_imputer.fit_transform(health_mode)


# **KNN Imputation**

# In[281]:


from fancyimpute import KNN # importing KNN library
knn_imputer = KNN() # calling KNN function
health_knn = Health_data.copy(deep=True) # creating copy
health_knn.iloc[:, :] = knn_imputer.fit_transform(health_knn)


# **Visualising Imputation**

# In[282]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 5)) # creating 2 rows and 3 columns
nullity = Health_data['age'].isnull() + Health_data['BMI'].isnull() # creating null columns between culmen length and culmen depth
imputations = {'Mean Imputation': health_mean, # creating a python dictionary
               'Median Imputation': health_median,
               'Most Frequent Imputation': health_mode,
               'KNN Imputation': health_knn}

for ax, df_key in zip(axes.flatten(), imputations): # a for loop to iterate over the subplots and the imputed data
    imputations[df_key].plot(x='age', y='BMI', kind='scatter',
                             alpha=0.5, c=nullity, cmap='rainbow', ax=ax,
                             colorbar=False, title=df_key)


# * we can see here that the distribution of data after KNN imputation have better distribution.
# * So , we will use KNN imputation in our dataset.

# In[283]:


Health_data = health_knn


# In[284]:


Health_data.isna().sum()


# In[285]:


print(Health_data['gender'].value_counts())
print(Health_data['Children'].value_counts())
print(Health_data['location'].value_counts())
print(Health_data['smoking_status'].value_counts())


# In[286]:


Health_data.describe()


# **storing Independent variables and dependent variables separately**

# In[287]:


#features Column
X = Health_data.drop(['health_insurance_price'] , axis = 1)


# In[288]:


X


# In[289]:


#Target/Response Variable
y = Health_data['health_insurance_price']
y


# **Observations**
# * we cannot see any high correlation between any two features.
# * So , we will not remove any feature
# 

# ### Train/Test Split

# * Splitting Dataset into two parts 
# * Train the Model on Training Set
# * Test the Model on Testing set

# In[290]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2 , random_state = 101) #keeping 20% data for testing , 80% for training.


# In[291]:


print(X.shape, X_train.shape, X_test.shape)


# In[292]:


print(X_train)


# In[293]:


print(y_train)


# **Feature Scaling**

# In[294]:


from sklearn.preprocessing import StandardScaler # performing standardization technique
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_train


# In[295]:


X_test=sc.transform(X_test)
X_test


# ## importing Models

# In[296]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# **Model Training**

# In[297]:


lr = LinearRegression()
lr.fit(X_train , y_train)
svm = SVR()
svm.fit(X_train , y_train)
rf = RandomForestRegressor()
rf.fit(X_train , y_train)
gr = GradientBoostingRegressor()
gr.fit(X_train , y_train)


# **Prediction on Test Data**

# In[298]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)

df1 = pd.DataFrame({'actual' : y_test, 'Lr' : y_pred1,
                  'svm' : y_pred2 , 'rf' : y_pred3 , 'gr' : y_pred4})


# In[299]:


df1


# ### Comparing Performance Visually

# In[300]:


plt.subplot(221)
plt.plot(df1['actual'].iloc[0 : 11],label = 'actual')
plt.plot(df1['Lr'].iloc[0:11] , label = 'Lr')
plt.legend()

plt.subplot(222)
plt.plot(df1['actual'].iloc[0 : 11],label = 'actual')
plt.plot(df1['svm'].iloc[0:11] , label = 'svm')
plt.legend()

plt.subplot(223)
plt.plot(df1['actual'].iloc[0 : 11],label = 'actual')
plt.plot(df1['rf'].iloc[0:11] , label = 'rf')
plt.legend()

plt.subplot(224)
plt.plot(df1['actual'].iloc[0 : 11],label = 'actual')
plt.plot(df1['gr'].iloc[0:11] , label = 'gr')

plt.tight_layout()
plt.legend()


# **Evaluating the Algorithm**

# we will evaluate our model using R2 , R2 is used to measure the goodness of fit , greater the value of R2 better the Regression model. Since we are having multiple linear regression problem , we can't use R2 value to evaluate accuracy of model , so we need to use adjusted R2 method.

# In[301]:


#evaluating using metrics
from sklearn import metrics


# In[302]:


r2m1 = metrics.r2_score(y_test , y_pred1)
r2m2 = metrics.r2_score(y_test , y_pred2)
r2m3 = metrics.r2_score(y_test , y_pred3)
r2m4 = metrics.r2_score(y_test , y_pred4)


# In[303]:


print(r2m1,r2m2,r2m3,r2m4)


# **Since we are having multiple linear regression problem , we can't use R2 value to evaluate accuracy of model , so we need to use adjusted R2 method**

# In[304]:


n= 1336   #number of observations
k=6  #number of independent variables
adj_r2_scorem1 = 1 - ((1-r2m1)*(n-1)/(n-k-1))
print(adj_r2_scorem1)

adj_r2_scorem2 = 1 - ((1-r2m2)*(n-1)/(n-k-1))
print(adj_r2_scorem2)

adj_r2_scorem3 = 1 - ((1-r2m3)*(n-1)/(n-k-1))
print(adj_r2_scorem3)

adj_r2_scorem4 = 1 - ((1-r2m4)*(n-1)/(n-k-1))
print(adj_r2_scorem4)


# **Observations :**
# 
# **we can see that Model 4 is performing better than any other Model , that is Gradient Boosting Algorithm**

# In[305]:


#evaluating using Mean Absolute Error
s1 = metrics.mean_absolute_error(y_test , y_pred1)
s2 = metrics.mean_absolute_error(y_test , y_pred2)
s3 = metrics.mean_absolute_error(y_test , y_pred3)
s4 = metrics.mean_absolute_error(y_test , y_pred4)


# In[306]:


print(s1,s2,s3,s4)


# **Observations :**
#     
# **Mean Absolute error is also used to measure the accuracy of Model**
# * Here , the lower is better.
# * we can see here also Model 4(Gradient Boosting Regression Model) is performing better among all Models .

# **Mean of residuals**
# 
# It should be close to zero

# In[307]:


residuals = y_test.values-y_pred1
mean_residuals = np.mean(residuals)
print("Mean of Residuals LR {}".format(mean_residuals))

residuals = y_test.values-y_pred2
mean_residuals = np.mean(residuals)
print("Mean of Residuals SVM {}".format(mean_residuals))

residuals = y_test.values-y_pred3
mean_residuals = np.mean(residuals)
print("Mean of Residuals RF {}".format(mean_residuals))

residuals = y_test.values-y_pred4
mean_residuals = np.mean(residuals)
print("Mean of Residuals GR {}".format(mean_residuals))


# **Homoscedasticity**

# In[308]:


p = sns.scatterplot(x=y_pred4, y=residuals)
plt.xlabel('y_pred4/predicted values')
plt.ylabel('Residuals')
p = sns.lineplot(x=[0,2.6],y=[0,0],color='Blue')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')


# In[309]:


#Errors should be normally distributed
sns.distplot((y_test-y_pred4),bins=50)


# **conclusion :**
# * Gradient Boosting Algorithm is giving best results.
# * So , we will use Gradient Boosting Algorithm for Model Deployment.

# ### Predicting house_insurance_price for new customer

# In[310]:


# Predictions from our Model
# Let's grab predictions off our test set and see how well it did!


predictions = gr.predict(X_test)


# In[311]:


plt.scatter(y_test,predictions)


# In[312]:


data = {'age' : 18.0,
        'gender' : 0 ,
        'BMI' : 31.92 ,
        'Children' : 0 ,
        'smoking_status' : 0,
        'location' : 4 }

df = pd.DataFrame(data , index = [0])
df


# In[313]:


new_pred = gr.predict(df)
print(new_pred)


# In[ ]:





# **1- Why is this proposal important in today’s world? How predicting a health insurance cost accurately can affect the health care/insurance field?**
# 

# Health insurance policy is an assurance which provides immediate financial help in case when any medical emergency arises. It is a contract between a policyholder and the insurance company which covers medical expenses that might occur due to illness, injury or accident. If you have a health insurance policy, then some or all the medical expenses will be borne by the insurance company, against which an insured is supposed to pay a certain amount known as premium.
# 
# Healthcare expenses are increasing at a rate higher than medical inflation, and that is why it is must for everyone to have a health insurance cover which not only helps you to save your emergency funds and saving of lifetime, in case any medical emergency occurs to you or your near and dear ones, but also supports you to deal with rising medical costs. 
# 
# Experts believe that a health insurance policy must be a part of your financial planning and it should be purchased early when you are young and responsible to stay safe and secured. Investing in a health insurance plan at an early age also provides other numerous advantages such as better sum insurance coverage, lower premium rates, no medical tests and so on.
# 
# Most of the health insurance providers have their own set parameters and based on them they fix the health insurance price. After conducting the research, compiling the historical data and analyzing your customer profile they decide to accept or reject your health insurance application. After assessing the risk factors, the health insurer will send the final quotation. It is important to know that every insurer uses its own assumptions and has its own set of standards while calculating the health insurance premium. 
# 
# It is very challenging for companies to decide the premium cost and also to accept or reject the application.
# As , there can be many applicants who can have pre-existing diseases , and if they fail to recognise this and have given insurance to them at low cost then there will be a huge loss for them.
# So, its very important to predict the premium cost as per the type and health status of applicants.
# Like , applicants who are smokers , are more likely to have health issues in future , so the premium cost to them must be higher. Premium cost also depends on Age.

# **2- If any, what is the gap in the knowledge, or how your proposed method can be helpful if required in the future for any other type of insurance?**

# For Insurance companies it is very difficult task to predict the Insurance Amount.
# This applies for insurance in every sector. for example : Bike insurance , car insurance , house insurance , death insurance etc.
# In every sector the problem is same , it is very challenging for companies to finalise the premium cost.
# It's must be not much higher as when it will be much higher , applicant will not take the policy , it will also be a loss for company.
# And if , there are pre-existing issues and insurance have been provided by company at lower amount , then it will claim and in this case also there is a loss for company.
# 
# Working on this project I have found some parameters on which health insurance price depends , and aable to build a predictor to predict the amount.
# with this knowledge I can build cost predictor for some other sectors also like automobile , life etc.
# 

# **3-  Please aim to identify patterns in the data and important features that may impact an ML model.**

# As per my analysis , 
# * The smoking status of applicant , have a huge impact on the health_insurance_price
# * Age is also a important factor , as the age increases the health_insurance_price also increases.
# 
