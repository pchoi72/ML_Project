#!/usr/bin/env python
# coding: utf-8

# ## Importing Data and Preliminary Exploration

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from scipy import stats
from scipy.stats import norm
import math
import sklearn

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[103]:


raw_data = pd.read_csv('../data/train.csv')
raw_data_test = pd.read_csv('../data/test.csv')


# In[3]:


raw_data.head(10)


# In[4]:


raw_data.columns


# In[5]:


raw_data.dtypes


# In[6]:


# Identify which columns have missing values
trainNA = raw_data.isnull().sum().sort_values(ascending = False)
trainNA_percent = (raw_data.isnull().sum()/len(raw_data)*100).sort_values(ascending=False)
missingness = pd.concat([trainNA, trainNA_percent], axis = 1, keys = ['Total', 'Percent'])
missingness[missingness['Total']>0]


# #### Psuedo-Missing
# 
# - PoolQC --> 'No Pool'
# - Alley --> 'No Alley'
# - Fence --> 'No Fence'
# - FireplaceQu --> 'No Fireplace'
# - MiscFeature --> 'None'
# - MasVnrType --> 'None'
# 
# *No Garage*
# - GarageCond
# - GarageFinish
# - GarageQual
# - GarageType
# - GarageYrBlt
# 
# *No Basement*
# - BsmtCond
# - BsmtQual
# - BsmtFinType1
# 
# 
# #### Actual Missing
# 
# - Electrical --> 'Mode'
# - MasVnrArea --> 'Mode' (group by Neighborhood & Year Built)
# - LotFrontage --> Impute with KNN based on LotConfig, LotShape, and LotArea (Both LotFrontage and missingness in LotFrontage is not evenly distributed across neighborhoods, so Median imputation is less valid)
# - BsmtExposure --> 'Mode' (thirty-seven rows had no values for this feature *or* any other basement feature; those were imputed to be "No Basement." One row had values for other basement columns but not this one; imputed via Mode (group by Neighborhood & Year Built))
# - BsmtFinType2 --> 'Mode' (see above comment)

# In[7]:


raw_data.groupby('Neighborhood').mean()['LotFrontage'].sort_values(ascending=False)


# In[8]:


raw_data[raw_data['LotFrontage'].isnull()].groupby('Neighborhood').count()['Id'].sort_values(ascending=False)


# ### Imputing Pseudo-Missingness

# In[14]:


def impute_pseudo(df):
    df['Alley'] = df['Alley'].fillna('No Alley')
    df['Fence'] = df['Fence'].fillna('No Fence')
    df['MiscFeature'] = df['MiscFeature'].fillna('None')
    df.loc[df['PoolArea']==0, 'PoolQC'] = 'No Pool'
    df.loc[np.logical_and(df['PoolArea']!=0, df['PoolQC'].isnull()==True), 'PoolQC'] = df['PoolQC'].mode()[0]
    df.loc[df['Fireplaces']==0, 'FireplaceQu'] = 'No Fireplace'
    df.loc[np.logical_and(df['Fireplaces']!=0, df['FireplaceQu'].isnull()==True), 'FireplaceQu'] = df['FireplaceQu'].mode()
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df.loc[np.logical_and(df['MasVnrArea']!=0, df['MasVnrType'].isnull()==True), 'MasVnrType'] = df['MasVnrType'].mode()[0]
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])


# In[15]:


# Create a function to impute all misssing values for features related to basements 
def impute_basements(df):
    col_list = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    num_col_list = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
    
    var_len = {}
    for i in col_list:
        if df[i].isnull().sum()>0:
            var_len[i] = df[i].isnull().sum()
    shortest_key = min(var_len, key = var_len.get)
    shortest_value = var_len[shortest_key]
    
    for i in col_list:
        if df[i].isnull().sum()==shortest_value:
            df[i] = df[i].fillna('No Basement')
            
    df.loc[df[shortest_key]=='No Basement', col_list] = "No Basement"
    df.loc[df[shortest_key]=='No Basement', num_col_list] = float(0)
    
    for i in col_list:
        if df[i].isnull().sum() > 0:
            df[i] = df[i].fillna(df[i].mode()[0])


# In[16]:


# Create a function to impute all misssing values for features related to garages 
def impute_garages(df):
    col_list = ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual']
    num_col_list = ['GarageYrBlt', 'GarageCars', 'GarageArea']
    
    var_len = {}
    for i in col_list:
        if df[i].isnull().sum()>0:
            var_len[i] = df[i].isnull().sum()
    shortest_key = min(var_len, key = var_len.get)
    shortest_value = var_len[shortest_key]
    
    for i in col_list:
        if df[i].isnull().sum()==shortest_value:
            df[i] = df[i].fillna('No Garage')
            
    df.loc[df[shortest_key]=='No Garage', col_list] = "No Garage"
    df.loc[df[shortest_key]=='No Garage', num_col_list] = float(0)
    
    for i in col_list:
        if df[i].isnull().sum() > 0:
            df[i] = df[i].fillna(df[i].mode()[0])


# In[1]:


from sklearn.neighbors import KNeighborsRegressor

def impute_lotfront(df):
    for config in df['LotConfig'].unique():
        X_train = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==False), ['LotArea']]
        X_fit = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==True), ['LotArea']]
        y_train = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==False), ['LotFrontage']]
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train, y_train)
        #df.loc[df['LotConfig']==config, 'LotFrontage'] = knn.predict(X_fit)
        print(config, knn.predict(X_fit).reshape(-1,1))
        print("*"*50)
        print(knn.predict(X_fit).shape)
        print(len(df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==True)]))
        print("*"*50)
        
        ### TO DO NOTE: Insert a try/except statement (or an if statement) to control for errors thrown
        ### if the LotConfig slice has no LotFrontage null values.


# In[112]:


impute_lotfront(raw_data)


# In[74]:


train_impute = raw_data.copy()
test_impute = raw_data_test.copy()


# In[75]:


impute_basements(train_impute)
impute_garages(train_impute)
impute_pseudo(train_impute)
impute_lotfront(train_impute)


# In[301]:


test_impute.isnull().sum()


# In[ ]:




