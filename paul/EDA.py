#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

##################################################
############### Data Pipeline ####################
##################################################
# See README file for explanation of philosophy and methodology

############### Import Data ###############
train_raw = pd.read_csv('../data/train.csv')
test_raw = pd.read_csv('../data/test.csv')


############### Remove Outliers ###############
# Create a function to remove outliers identified in EDA
def remove_outliers(df):
    df =df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
    return df


############### Impute Data ###############
# Create a function handles imputation of most pseudo-missing and missing values (other than the features related to basements, lot frontage, and garages)
def impute_pseudo(df):
    # Impute pseudo-missing values
    df['Alley'] = df['Alley'].fillna('No Alley')
    df['Fence'] = df['Fence'].fillna('No Fence')
    df['MiscFeature'] = df['MiscFeature'].fillna('None')
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    # Some variables in pseudo-missing columns actually represent missing observations (as can be told from feature variables relating to the same category of housing feature)
    # Impute as the mode if the missing value represents a missing observation, impute as "No X" if it does not
    df.loc[df['PoolArea']==0, 'PoolQC'] = 'No Pool'
    df.loc[np.logical_and(df['PoolArea']!=0, df['PoolQC'].isnull()==True), 'PoolQC'] = df['PoolQC'].mode()[0]
    df.loc[df['Fireplaces']==0, 'FireplaceQu'] = 'No Fireplace'
    df.loc[np.logical_and(df['Fireplaces']!=0, df['FireplaceQu'].isnull()==True), 'FireplaceQu'] = df['FireplaceQu'].mode()
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df.loc[np.logical_and(df['MasVnrArea']!=0, df['MasVnrType'].isnull()==True), 'MasVnrType'] = df['MasVnrType'].mode()[0]
    # Impute true missing values with mode imputation
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])


# Create a function to impute misssing values for the feature variables related to basements.
def impute_basements(df):
    col_list = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    num_col_list = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
    
    #Generate a dictionary to calculate which basement feature has the fewest number of missing observations
    var_len = {}
    for i in col_list:
        if df[i].isnull().sum()>0:
            var_len[i] = df[i].isnull().sum()
    shortest_key = min(var_len, key = var_len.get)
    shortest_value = var_len[shortest_key]
    
    # Encode all observations that are missing the same number of observations as the least missing feature as "No Basement"
    for i in col_list:
        if df[i].isnull().sum()==shortest_value:
            df[i] = df[i].fillna('No Basement')
    
    # If a given feature is missing and that observation has been established as having No Basement above, impute as "No Basement"
    # for categorical features or 0 for numeric features.        
    df.loc[df[shortest_key]=='No Basement', col_list] = "No Basement"
    df.loc[df[shortest_key]=='No Basement', num_col_list] = float(0)
    
    # If a given feature is missing and that observation has not already been established as No Basement or 0 above, that feature represents
    # true missingness. Fill any missing values with the mode of that feature
    for i in col_list:
        if df[i].isnull().sum() > 0:
            df[i] = df[i].fillna(df[i].mode()[0])


# Create a function to impute missing values for the feature variables related to garages.
def impute_garages(df):
    col_list = ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual']
    num_col_list = ['GarageYrBlt', 'GarageCars', 'GarageArea']
    
    #Generate a dictionary to calculate which garage feature has the fewest number of missing observations
    var_len = {}
    for i in col_list:
        if df[i].isnull().sum()>0:
            var_len[i] = df[i].isnull().sum()
    shortest_key = min(var_len, key = var_len.get)
    shortest_value = var_len[shortest_key]
    
    # Encode all observations that are missing the same number of observations as the least missing feature as "No Garage"
    for i in col_list:
        if df[i].isnull().sum()==shortest_value:
            df[i] = df[i].fillna('No Garage')
    
    # If a given feature is missing and that observation has been established as having No Garage above, impute as "No Garage"
    # for categorical features or 0 for numeric features.
    df.loc[df[shortest_key]=='No Garage', col_list] = "No Garage"
    df.loc[df[shortest_key]=='No Garage', num_col_list] = float(0)
    
    # If a given feature is missing and that observation has not already been established as No Garage or 0 above, that feature represents
    # true missingness. Fill any missing values with the mode of that feature
    for i in col_list:
        if df[i].isnull().sum() > 0:
            df[i] = df[i].fillna(df[i].mode()[0])
            
    # Impute any remaining missing values in the Garage features
    df['GarageCars'] = df.groupby(['GarageType'], sort=False)['GarageCars'].apply(lambda x: x.fillna(round(x.mean(), 0)))
    df['GarageArea'] = df.groupby(['GarageType'], sort=False)['GarageArea'].apply(lambda x: x.fillna(round(x.mean(), 0)))
    df['GarageYrBlt'] = df.groupby(['YearBuilt'], sort=False)['GarageYrBlt'].apply(lambda x: x.fillna(round(x.mean(), 0)))


# Create a function to impute misssing values for the LotFrontage feature variable.
def impute_lotfront(df):
    for config in df[df['LotFrontage'].isnull()]['LotConfig'].unique():
        X_train = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==False), ['LotArea']]
        X_fit = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==True), ['LotArea']]
        y_train = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==False), ['LotFrontage']]
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train, y_train)
        df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==True),'LotFrontage'] = knn.predict(X_fit)


# Create a function to re-engineer categorical feature variables for ease of dummification.
def impute_categorical(df):
    df['Condition1'] = df['Condition1'].apply(lambda x: "Norm" if x == "Norm" else "Other")
    df['LotShape'] = df['LotShape'].apply(lambda x: "Reg" if x == "Reg" else "IReg")
    df['FireplaceQu'] = df['FireplaceQu'].apply(lambda x: "No Fireplace" if x== 'No Fireplace' else "Fireplace")
    df['Functional'] = df['Functional'].apply(lambda x: "Y" if x=="Y" else "N")
    df['Electrical'] = df['Electrical'].apply(lambda x: "SBrkr" if x=='SBrkr' else 'Other')
    df['RoofMatl'] = df['RoofMatl'].apply(lambda x: "Other" if x != "CompShg" else x)
    df['RoofStyle'] = df['RoofStyle'].apply(lambda x: "Other" if (x !="Gable" and x != "Hip") else x)
    df['Heating'] = df['Heating'].apply(lambda x: "GasA" if x == "GasA" else 'Other')
    df['Foundation'] = df['Foundation'].apply(lambda x: "Other" if (x !="PConc" and x != "CBlock" and x != "BrkTil") else x)
    df['SaleType'] = df['SaleType'].apply(lambda x: "Other" if (x !="WD" and x != "New" and x != "COD") else x)
    df['Exterior1st'] = df['Exterior1st'].apply(lambda x: "Other" if (
        x !="VinylSd" and x != "MetalSd" and x != "HdBoard" and x != "Wd Sdng" and x != "Plywood") else x)
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
    df['Utilities'] = df['Utilities'].fillna('AllPub')
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])


# Create a function that combines all the above functions to perform all imputation at once.
def impute_data(df):
    impute_pseudo(df)
    impute_basements(df)
    impute_garages(df)
    impute_lotfront(df)
    impute_categorical(df)


############### Feature Engineering/Creation ###############
# Create a function to add new feature variables based on variables already in the data
def add_features(df):
    df['IsPool']=df['PoolQC'].apply(lambda x: 1 if x !="No Pool" else 0)
    df['IsGarage']=df['GarageYrBlt'].apply(lambda x: 0 if x==0 else 1)
    df['TotalFullBath'] = df['BsmtFullBath'] + df['FullBath']
    df['TotalHalfBath'] = df['BsmtHalfBath'] + df['HalfBath']
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']


############### Feature Dummification ###############
# Create a function to dummify the categorical variables for use in regression
def dummify_features(df):
    # Dummify features related to what type of porch (if any) the house had by hard encoding zeros and ones
    df['3SsnPorch'] = df['3SsnPorch'].apply(lambda x: 1 if x>0 else 0)
    df['ScreenPorch'] = df['ScreenPorch'].apply(lambda x: 1 if x>0 else 0)
    df['EnclosedPorch'] = df['EnclosedPorch'].apply(lambda x: 1 if x>0 else 0)
    df['IsOpenPorch'] = df['OpenPorchSF'].apply(lambda x: 1 if x>0 else 0)
    # Dummify all other necessary features using One Hot Encoding
    temp = df.copy()
    col_list = ['MoSold', 'YrSold', 'OverallQual', 'OverallCond', 'Exterior1st', 'Condition1', 'LotShape', 'FireplaceQu',
    'Functional', 'Electrical', 'RoofMatl', 'RoofStyle', 'Heating', 'Foundation', 'SaleType', "LandContour", 'MSZoning',
    'Street', 'Alley', 'HouseStyle', 'BldgType', 'LandSlope', 'LotConfig', 'Neighborhood', 'ExterCond', 'ExterQual',
    'GarageType', 'PavedDrive', 'KitchenQual', 'Fence', 'MasVnrType', 'CentralAir', 'SaleCondition', 'HeatingQC', 
    'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'GarageFinish', 'GarageCond']
    ohe = OneHotEncoder(categories = 'auto', drop = None, sparse = False)
    enc = ohe.fit_transform(temp[col_list])
    enc = pd.DataFrame(enc, columns=ohe.get_feature_names(col_list))
    temp = pd.concat((temp.drop(col_list, axis=1).reset_index(drop=True), enc), axis=1)
    return temp


############### Feature Removal ###############
# Create a function to remove features that are either not useful for prediction (ex. 'Id') or have been rendered redundant by the above steps
def remove_features(df):
    df = df.drop(columns=['Id','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
            "BsmtHalfBath",'FullBath','HalfBath','OpenPorchSF',"Condition2", "Utilities", "Exterior2nd", "GarageQual",
                          "PoolQC", "MiscFeature","BsmtFinType2"])
    return df


############### Data Processing and Saving Files ###############
# Create a function that handles all data processing steps above at once:
def process_data(df):
    try:
        df=remove_outliers(df)
    except:
        pass
    impute_data(df)
    add_features(df)
    df = dummify_features(df)
    df = remove_features(df)
    return df

# Create copies of the raw data and process them
train_processed = train_raw.copy() 
train_processed = process_data(train_processed)

test_processed = test_raw.copy()
test_processed = process_data(test_processed)

# Save the processed datasets to CSV files
# train_processed.to_csv('../data/train_processed.csv')
# test_processed.to_csv('../data/test_processed.csv')


# #### Variables Added
# List of variables that had to be added to the preprocessing function because they were missing in the test dataset but not the train dataset
# 
# - 'MSZoning'    4 missing values. Categorical. 5 possible values. Mode imputation
# 
# -  'Utilities'  2 missing values. Categorical. 1 possible value ('AllPub'), so impute as that
# 
# - 'Exterior2nd' 1 missing value. Categorical. 15 possible values. Mode imputation
# 
# - 'KitchenQual' 1 missing value. Categorical. 4 possible values (TA, Gd, Ex, Fa). Mode imputation
# 
# - 'GarageYrBlt' 2 missing values. Numeric (tons of possible values). Mode imputation grouped by HouseBuilt
# 
# - 'GarageCars' 1 missing value. Numeric. Mode imputation grouped by GarageType
# 
# - 'GarageArea' 1 missing value. Numeric (many possible values, all very disparate). Mode imputation grouped by GarageType

# # Exploratory Data Analysis & Visualization
# 
# Below we examine the raw data from both the `train` and `test` datasets.

# In[2]:


#Loading packages for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
from scipy import stats
from scipy.stats import norm
import math


# In[3]:


#Checking the shape for both datasets
print(train_raw.shape)
print(test_raw.shape)


# In[4]:


# Quick heatmap visualization to see correlation among the columns
train_raw.drop(['Id'], axis=1, inplace=True)
fig, ax = plt.subplots(figsize=(11.5,8.5))
sns.heatmap(train_raw.corr(),square=True)


# In[5]:


#Visualization to quickly see missing values in each column
fig, ax = plt.subplots(figsize=(10,8))
sns.set_style("white")
sns.heatmap(train_raw.isnull(), cbar=False)


# In[6]:


#Table to see the sum of missing values and percent of total observations
train_NAs = train_raw.isnull().sum().sort_values(ascending=False)
train_percent_NAs = (train_raw.isnull().sum()/train_raw.isnull().count()*100).sort_values(ascending=False)
missing = pd.concat([train_NAs, train_percent_NAs], axis=1, keys=['Total', 'Percent'])
missing.head(20)


# In[7]:


plt.style.use('ggplot')
#Chart to show total missing values
train_NAs.nlargest(20).plot(kind='bar', figsize=(10,5), title='Total observations missing by feature')


# In[8]:


#Chart to see the percent of missing values
train_percent_NAs.nlargest(20).plot(kind='bar', figsize=(10,5), title='Percent missing by feature')


# ### Examining the target variable: `SalePrice`

# In[9]:


#Plots to see SalePrice distribution
#sns.distplot(train_raw['SalePrice'], fit=norm).set(title='SalePrice Distribution',ylabel='Frequency')
sns.distplot(train_raw['SalePrice']).set(title='SalePrice Distribution',ylabel='Frequency')

#Get the fitted parameters by function
(mu, sigma) = norm.fit(train_raw['SalePrice'])
print('mu = {:.2f}'.format(mu))
print('sigma = {:.2f}'.format(sigma))


# In[10]:


stats.probplot(train_raw['SalePrice'], plot=plt)


# In[11]:


#Plot to see relationship between SalePrice and Above Ground Living Area (in square feet)
sns.jointplot(x="GrLivArea", y="SalePrice", data=train_raw)


# In[15]:


n = 46
plt.scatter(train_raw.iloc[:,n],train_raw['SalePrice'])
plt.xlabel(train_raw.columns[n])
plt.ylabel('SalePrice')


# ### Applying log tranformation to the target variable: `SalePrice`

# In[16]:


train_raw['SalePrice'] = np.log1p(train_raw['SalePrice'])
sns.distplot(train_raw['SalePrice'], fit=norm).set(title='SalePrice Distribution',ylabel='Frequency')
(mu, sigma) = norm.fit(train_raw['SalePrice'])
print('mu = {:.2f}'.format(mu))
print('sigma = {:.2f}'.format(sigma))


# In[17]:


fig = plt.figure()
ax = fig.add_subplot()
stats.probplot(np.log1p(train_raw['SalePrice']), plot=plt)
ax.set_title("Log Transformation SalePrice")


# In[18]:


fig = plt.figure()
ax = fig.add_subplot()
stats.probplot(stats.boxcox(train_raw['SalePrice'], lmbda = 0.3), plot=plt)
ax.set_title("BoxCox Transformation SalePrice")


# In[22]:


train_raw.hist(figsize=(20,18))


# In[ ]:




