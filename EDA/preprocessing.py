import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
import re
from scipy import stats
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

train_raw = pd.read_csv('../data/train.csv')
test_raw = pd.read_csv('../data/test.csv')

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

from sklearn.neighbors import KNeighborsRegressor

def impute_lotfront(df):
    
    for config in df[df.LotFrontage.isnull()]['LotConfig'].unique():
        X_train = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==False), ['LotArea']]
        X_fit = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==True), ['LotArea']]
        y_train = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==False), ['LotFrontage']]
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train, y_train)
        df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==True),'LotFrontage'] = knn.predict(X_fit)

def impute_data(df): 
    impute_pseudo(df)
    impute_basements(df)
    impute_garages(df)
    impute_lotfront(df)

