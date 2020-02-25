#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
import re
from scipy import stats
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[5]:


train_raw = pd.read_csv('../data/train.csv')
test_raw = pd.read_csv('../data/test.csv')


# In[7]:


train_raw.HouseStyle.value_counts()


# In[8]:


test_raw.HouseStyle.value_counts()


# In[156]:


from __future__ import print_function  # Python 2 and 3
from IPython.display import display
pd.options.display.max_columns = 100
pd.options.display.max_rows = 500


# In[162]:


train_raw[['SalePrice']].hist()
np.log(train_raw['SalePrice']).hist()


# In[163]:


# Identify which columns have missing values
trainNA = train_raw.isnull().sum().sort_values(ascending = False)
trainNA_percent = (train_raw.isnull().sum()/len(train_raw)*100).sort_values(ascending=False)
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
# 
# #### Ord Categorical & Numeric Features
# 
#  - Id               float64 --- Drop
#  - BsmtUnfSF        float64 --- Drop
#  - TotalBsmtSF      float64 --- Drop, re-engineered.
#  - SalePrice        float64 --- Target, Drop
#  - 1stFlrSF         float64 --- Drop, re-engineered.
#  - 2ndFlrSF         float64 --- Drop, re-engineered.
#  - LowQualFinSF     float64 --- Drop, 99% has value 0
#  - GrLivArea        float64 --- Drop, correlated with total SF and rooms above ground
#  - BsmtFullBath     float64 --- Drop, re-engineered.
#  - BsmtHalfBath     float64 --- Drop, re-engineered.
#  - FullBath         float64 --- Drop, re-engineered.
#  - HalfBath         float64 --- Drop, re-engineered.
#  - BedroomAbvGr     float64
#  - KitchenAbvGr     float64
#  - TotRmsAbvGrd     float64
#  - Fireplaces       float64
#  - GarageYrBlt      float64 --- Drop, correlated with YrBuilt
#  - GarageCars       float64 --- Drop, correlated with Area
#  - GarageArea       float64 
#  - WoodDeckSF       float64 
#  - OpenPorchSF      float64 --- dummify 1 for OpenPorch 0 for Not OpenPorch
#  - EnclosedPorch    float64 --- dummify if porch 1, else 0
#  - 3SsnPorch        float64 --- dummify if porch 1, else 0
#  - ScreenPorch      float64 --- dummify if porch 1, else 0
#  - PoolArea         float64 --- Drop, over 99% do not have pools
#  - MiscVal          float64 --- **add 1 and then Log transform 
#  - MoSold           float64 --- dummify
#  - YrSold           float64 --- dummify
#  - BsmtFinSF2       float64 --- Drop
#  - BsmtFinSF1       float64 --- Drop
#  - IsPool           float64 --- Categorical
#  - MSSubClass       float64 --- Categorical - convert back to Str
#  - LotFrontage      float64
#  - MasVnrArea       float64
#  - OverallQual      float64 --- dummify
#  - LotArea          float64 --- **add 1 and then Log transform 
#  - OverallCond      float64 --- dummify
#  - YearBuilt        float64
#  - YearRemodAdd     float64
#  
#  
#  ###### New Features
#  
#  - Total SF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF
#  - TotalFullBath = BsmtFullBath + FullBath
#  - TotalHalfBath = BsmtHalfBath + HalfBath
#  - IsPool  --- Categorical
#  - IsGarage  --- Categorical
#  
#  
#  #### Outliers
#  
#  - 2 observations with large living Area but extremely low price. 
#      df =df.drop(df[(train['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
#  - 
# 
# 
# #### Categorical Dummify Var
# 
#  - LandContour        object --- 4 values, dummify
#  - Exterior1st        object --- 14 values, **Top5(VinylSd, MetalSd, HdBoard, Wd Sdng, Plywood) or Other.
#  - RoofMatl           object --- Dummify. **CompShg or Other
#  - RoofStyle          object --- Dummify. **Gable, Hip or Other.
#  - MSZoning           object --- Dummify.
#  - Street             object --- Dummify. 2 values
#  - Alley              object --- Dummify. 3 values
#  - HouseStyle         object --- Dummify. 8 values
#  - BldgType           object --- Dummify. 5 values
#  - Condition2         object --- Drop, redundant with Condition1
#  - Condition1         object --- Dummify. **Norm or Other.
#  - Exterior2nd        object --- Drop, redundant with Exterior1, 
#  - LandSlope          object --- Dummify. 3 values
#  - LotConfig          object --- Dummify. 3 values
#  - Utilities          object --- Drop. 1 observation 
#  - LotShape           object --- Dummify. **Reg or IReg
#  - Neighborhood       object --- Dummify.
#  - ExterCond          object --- Dummify.
#  - ExterQual          object --- Dummify.
#  - GarageType         object --- Dummify.
#  - GarageQual         object --- Drop, redundant with GarageCond.
#  - FireplaceQu        object --- Dummify. **FirePlace or No Fireplace
#  - PavedDrive         object --- Dummify.
#  - Functional         object --- Dummify. **Y or N(P will be N)
#  - KitchenQual        object --- Dummify.
#  - PoolQC             object --- Drop. IsPool col is created.
#  - Fence              object --- Dummify.
#  - MiscFeature        object --- Drop. redundant with MiscValue
#  - Electrical         object --- Dummify. **SBrkr or Other
#  - MasVnrType         object --- Dummify.
#  - CentralAir         object --- Dummify.
#  - SaleCondition      object --- Dummify.
#  - HeatingQC          object --- Dummify.
#  - Heating            object --- Dummify. **GasA or Other
#  - BsmtFinType2       object --- Drop.
#  - BsmtFinType1       object --- Dummify.
#  - BsmtExposure       object --- Dummify.
#  - BsmtCond           object --- Dummify.
#  - BsmtQual           object --- Dummify.
#  - Foundation         object --- Dummify. **PConc, CBlock, BrkTil, Other
#  - GarageFinish       object --- Dummify. 
#  - SaleType           object --- Dummify. **WD, New, COD, Other
#  - GarageCond         object --- Dummify.
# 

# In[232]:


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


# In[164]:


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


# In[165]:


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


# In[166]:


from sklearn.neighbors import KNeighborsRegressor

def impute_lotfront(df):
    
    for config in df[df.LotFrontage.isnull()]['LotConfig'].unique():
        X_train = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==False), ['LotArea']]
        X_fit = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==True), ['LotArea']]
        y_train = df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==False), ['LotFrontage']]
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train, y_train)
        df.loc[np.logical_and(df['LotConfig']==config, df['LotFrontage'].isnull()==True),'LotFrontage'] = knn.predict(X_fit)


# In[277]:


def impute_categorical(df):
    df.Condition1 = df.Condition1.apply(lambda x: "Norm" if x == "Norm" else "Other")
    df.LotShape = df.LotShape.apply(lambda x: "Reg" if x == "Reg" else "IReg")
    df.FireplaceQu = df.FireplaceQu.apply(lambda x: "No Fireplace" if x== 'No Fireplace' else "Fireplace")
    df.Functional = df.Functional.apply(lambda x: "Y" if x=="Y" else "N")
    df.Electrical = df.Electrical.apply(lambda x: "SBrkr" if x=='SBrkr' else 'Other')
    df['RoofMatl'] = df['RoofMatl'].apply(lambda x: "Other" if x != "CompShg" else x)
    df['RoofStyle'] = df['RoofStyle'].apply(lambda x: "Other" if (x !="Gable" and x != "Hip") else x)
    df.Heating = df.Heating.apply(lambda x: "GasA" if x == "GasA" else 'Other')
    df['Foundation'] = df['Foundation'].apply(lambda x: "Other" if (x !="PConc" and x != "CBlock" and x != "BrkTil") else x)
    df['SaleType'] = df['SaleType'].apply(lambda x: "Other" if (x !="WD" and x != "New" and x != "COD") else x)
    df['Exterior1st'] = df['Exterior1st'].apply(lambda x: "Other" if (
        x !="VinylSd" and x != "MetalSd" and x != "HdBoard" and x != "Wd Sdng" and x != "Plywood") else x)


# In[276]:


def impute_data(df):
    impute_pseudo(df)
    impute_basements(df)
    impute_garages(df)
    impute_lotfront(df)
    impute_categorical(df)


# In[168]:


def add_features(df):
    df['IsPool']=df.PoolQC.apply(lambda x: 1 if x !="No Pool" else 0)
    df['IsGarage']=df.GarageYrBlt.apply(lambda x: 0 if x==0 else 1)
    df['TotalFullBath'] = df['BsmtFullBath'] + df['FullBath']
    df['TotalHalfBath'] = df['BsmtHalfBath'] + df['HalfBath']
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']


# In[267]:


def dummify_features(df):
     # porch
    df['3SsnPorch'] = df['3SsnPorch'].apply(lambda x: 1 if x>0 else 0)
    df['ScreenPorch'] = df['ScreenPorch'].apply(lambda x: 1 if x>0 else 0)
    df['EnclosedPorch'] = df['EnclosedPorch'].apply(lambda x: 1 if x>0 else 0)
    df['IsOpenPorch'] = df['OpenPorchSF'].apply(lambda x: 1 if x>0 else 0)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    copy = df.copy()
    ohe = OneHotEncoder(categories = 'auto', drop = None, sparse = False)
    enc = ohe.fit_transform(copy[['MoSold','YrSold','OverallQual','OverallCond','Exterior1st','Condition1',
                                  'LotShape','FireplaceQu','Functional','Electrical','RoofMatl','RoofStyle',
                                  'Heating','Foundation','SaleType',"LandContour",'MSZoning','Street','Alley',
                                  'HouseStyle','BldgType','LandSlope',
                                 'LotConfig','Neighborhood','ExterCond','ExterQual','GarageType','PavedDrive',
                                 'KitchenQual','Fence','MasVnrType','CentralAir','SaleCondition','HeatingQC',
                                 'BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','GarageFinish','GarageCond']])
    enc = pd.DataFrame(enc, columns=ohe.get_feature_names(['MoSold','YrSold','OverallQual','OverallCond','Exterior1st','Condition1','LotShape','FireplaceQu','Functional','Electrical','RoofMatl','RoofStyle','Heating','Foundation','SaleType',"LandContour",'MSZoning','Street','Alley','HouseStyle','BldgType','LandSlope',
                                 'LotConfig','Neighborhood','ExterCond','ExterQual','GarageType','PavedDrive',
                                 'KitchenQual','Fence','MasVnrType','CentralAir','SaleCondition','HeatingQC',
                                 'BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','GarageFinish','GarageCond']))
    copy = pd.concat((copy.drop(['MoSold','YrSold','OverallQual','OverallCond','Exterior1st','Condition1','LotShape','FireplaceQu','Functional','Electrical','RoofMatl','RoofStyle','Heating','Foundation','SaleType',"LandContour",'MSZoning','Street','Alley','HouseStyle','BldgType','LandSlope',
                                 'LotConfig','Neighborhood','ExterCond','ExterQual','GarageType','PavedDrive',
                                 'KitchenQual','Fence','MasVnrType','CentralAir','SaleCondition','HeatingQC',
                                 'BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','GarageFinish','GarageCond'], 
                                axis=1).reset_index(drop=True), enc), axis=1)
    return copy


# In[294]:


def remove_features(df):
    df = df.drop(columns=['Id','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
            "BsmtHalfBath",'FullBath','HalfBath','OpenPorchSF',"Condition2", "Utilities", "Exterior2nd", "GarageQual",
                          "PoolQC", "MiscFeature","BsmtFinType2"])
    return df


# In[171]:


def remove_outliers(df):
    df =df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
    return df


# In[ ]:


def transform_data(df, isTraining=False):
    if isTraining ==True:
        df=remove_outliers(df)
    impute_data(df)
    add_features(df)
    df=dummify_features(df)
    df=remove_features(df)
    return df


# In[295]:


train_impute = train_raw.copy()
test_impute = test_raw.copy()
train_impute=remove_outliers(train_impute)
impute_data(train_impute)
add_features(train_impute)
train_impute=dummify_features(train_impute)
train_impute=remove_features(train_impute)


# In[314]:


# pd.set_option('display.max_columns', None)
# train_impute.dtypes


# In[2]:


train_raw.columns


# In[360]:


diff = []
for i in missing_test_col:
    if i not in missing_train_col:
        diff.append(i)


# In[366]:


diff


# In[364]:


diff2 = []
for i in missing_train_col:
    if i not in missing_test_col:
        diff2.append(i)


# In[209]:


intCol=train_impute.columns[train_impute.dtypes=='int']
dumCol = train_impute.columns[train_impute.dtypes=='uint8']
for col in intCol:
    train_impute[col]=train_impute[col].astype('float')


# In[212]:


a=[train_impute[train_impute.columns[train_impute.isnull().any(axis=0)==False]].dtypes=='float'][0] 
numCol = a[a==True].index # numeric columns with no missing values
olsCol=numCol.to_list()
olsCol.extend(dumCol.to_list())


# In[216]:


olsFeature=train_impute[olsCol]
target = train_impute['SalePrice']
# pd.DataFrame.corr(numFeature)


# In[370]:


y = train_impute['SalePrice'] # target variable
X = train_impute.loc[:, train_impute.columns!='SalePrice']
X = X.loc[:, X.columns.any(diff)]
print(X.shape, y.shape)


# In[1]:


from sklearn.model_selection import train_test_split
np.random.seed(1)
x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=0.25)
ridge = linear_model.Ridge()
ridge.set_params(alpha = 0, normalize = True)
ridge.fit(x_tr, y_tr)
ridge.score(x_tr, y_tr)


# In[300]:


ridge.score(x_val, y_val)


# In[308]:


test_raw


# In[307]:


from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
lm = LinearRegression()
lm.fit(x_train, y_train)
print(lm.score(x_train, y_train))

lm.fit(x_test, y_test)
print(lm.score(x_test, y_test))


# In[228]:


ridge.get_params()


# In[229]:


from sklearn.model_selection import GridSearchCV #Cross Validation
params = [{'alpha':[0.01, 0.1, 1, 10, 100]}]
gridSearch=GridSearchCV(estimator=ridge, param_grid=params)
gridSearch.fit(x_tr, np.log(y_tr))
gridSearch.best_params_


# In[230]:


ridge_best = linear_model.Ridge()
ridge_best.set_params(alpha = 0.1, normalize = True)
ridge_best.fit(x_tr, np.log(y_tr))
ridge_best.score(x_tr, np.log(y_tr))


# In[231]:


ridge_best.score(x_val, np.log(y_val))


# In[ ]:


testFeatures=


# In[413]:


Predict_Price = np.exp(ridge_best.predict(testFeatures))

pred = pd.DataFrame(Predict_Price, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('submission_1.csv',index=False)


# In[127]:


pd.merge(train_raw, pd.get_dummies(train_raw.MoSold, prefix='Mon'), left_index=True, right_index=True).drop(columns=['Mon_1'])


# In[295]:


# how do different coefficients shrink?
alphas = np.arange(0,100,10)
ridge.set_params(normalize=True)
coefs  = []
scores = []
for alpha in alphas:
        ridge.set_params(alpha=alpha)
        ridge.fit(X_train, np.log(y_train))
        coefs.append(ridge.coef_)
        scores.append(ridge.score(X_train, np.log(y_train)))
coefs = pd.DataFrame(coefs, index = alphas, columns = X_train.columns)  
coefs


# In[296]:


plt.rcParams['figure.figsize'] = (10,5)
for name in coefs.columns:
    plt.plot(coefs.index, coefs[name], label=name)
plt.legend(loc=4)   
plt.xlabel(r'hyperparameter $\lambda$')
plt.ylabel(r'slope values')


# In[297]:


plt.plot(alphas, scores, c='b', label=r'$R^2$')
plt.legend(loc=1)
plt.title(r'$R^2$ Drops with Increaing Regularizations')
plt.xlabel(r'hyperparameter $\lambda$')
plt.ylabel(r'$R^2$')


# In[298]:


ridge.set_params(normalize=True)
ridge_scores_train = []
ridge_scores_test  = []

alphas = np.linspace(53, 55, 100)

for alpha in alphas:
            ridge.set_params(alpha=alpha)
            ridge.fit(X_train, np.log(y_train))
            ridge_scores_train.append(ridge.score(X_train, np.log(y_train)))
            ridge_scores_test.append(ridge.score(X_test, np.log(y_test)))
ridge_scores_train = np.array(ridge_scores_train) 
ridge_scores_test  = np.array(ridge_scores_test)


# In[299]:


plt.plot(alphas, ridge_scores_train, label=r'$train\ R^2$')
plt.plot(alphas, ridge_scores_test, label=r'$test\ R^2$')
plt.legend(loc=1)
plt.title(r'Ridge Train-Test $R^2$ Comparison')
ridge_underfit = ridge_scores_train < ridge_scores_test
last_underfit  = np.max(alphas[ridge_underfit])
plt.axvline(last_underfit, linestyle='--', color='g', label='optimal lambda', alpha=0.4)
plt.legend(loc=1)
plt.xlabel(r'hyperparameter $\lambda$')
plt.ylabel(r'$R^2$')


# In[ ]:




