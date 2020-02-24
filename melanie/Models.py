#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import re
from scipy import stats
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[20]:


from __future__ import print_function  # Python 2 and 3
from IPython.display import display
pd.options.display.max_columns = 100
pd.options.display.max_rows = 500


# In[7]:


train = pd.read_csv('../data/train_processed.csv')
test_raw = pd.read_csv('../data/test.csv')
test = pd.read_csv('../data/test_processed.csv')


# In[8]:


y = train['SalePrice'] # target variable
X = train.loc[:, train.columns!='SalePrice']
X = X.loc[:, X.columns!='HouseStyle_2.5Fin'] #HouseStyle_2.5Fin in training.csv but not in test.csv


# In[9]:


for i in test.columns:
    if i not in X.columns:
        print(i)


# In[10]:


test.shape


# In[11]:


X.shape


# ### OLM - all features with Log(y)

# In[187]:


np.random.seed(1)
x_tr, x_val, y_tr, y_val = train_test_split(X, np.log(y), test_size=0.25)
olm = linear_model.LinearRegression()
olm.fit(x_tr, y_tr)
print(olm.score(x_tr, y_tr))
print(olm.score(x_val, y_val))


# In[58]:


prediction=np.exp(olm.predict(test))
prediction


# In[63]:


pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/submission_2.csv',index=False)


# ### OLM - all features with BoxCox on y

# In[172]:


from scipy.stats import boxcox
yt = stats.boxcox(y, lmbda = 0.05)


# In[186]:


from sklearn.model_selection import train_test_split
np.random.seed(1)
x_tr, x_val, y_tr, y_val = train_test_split(X, yt, test_size=0.25)
olm = linear_model.LinearRegression()
olm.fit(x_tr, y_tr)
print(olm.score(x_tr, y_tr))
print(olm.score(x_val, y_val))


# In[176]:


from scipy.special import inv_boxcox
prediction=olm.predict(test)
prediction=inv_boxcox(prediction, 0.05)

pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/submission_3.csv',index=False)


# ### How do the coefficients shrink?

# In[35]:


# how do different coefficients shrink?
from sklearn.linear_model import Ridge
ridge = Ridge(normalize=True)
alphas = np.linspace(1e-3,1,100)
ridge.set_params(normalize=True)
coefs  = []
scores = []
for alpha in alphas:
        ridge.set_params(alpha=alpha)
        ridge.fit(X, np.log(y))
        coefs.append(ridge.coef_)
        scores.append(ridge.score(X, np.log(y)))
coefs = pd.DataFrame(coefs, index = alphas, columns = X.columns)  

plt.rcParams['figure.figsize'] = (20,10)
for name in coefs.columns:
    plt.plot(coefs.index, coefs[name], label=name)
plt.legend(loc=4)   
plt.xlabel(r'hyperparameter $\lambda$')
plt.ylabel(r'slope values')


# In[36]:





# ### Ridge - all features

# In[131]:


from sklearn.linear_model import Ridge
np.random.seed(1)
x_tr, x_val, y_tr, y_val = train_test_split(X, np.log(y), test_size = 0.2)
ridge = Ridge(normalize=True)
ridge.fit(x_tr, y_tr)
print(ridge.score(x_tr, y_tr))
print(ridge.score(x_val, y_val))

from sklearn.model_selection import GridSearchCV #Cross Validation
ridge = Ridge(normalize=True)
alphaRange = np.linspace(1e-3,1,100).tolist()
params = [{'alpha':alphaRange}]
gridSearch=GridSearchCV(estimator=ridge, param_grid=params)
gridSearch.fit(x_tr, y_tr)
print(gridSearch.best_params_)
ridge = Ridge(normalize=True,alpha=list(gridSearch.best_params_.values())[0])
ridge.fit(x_tr, y_tr)
print(ridge.score(x_tr, y_tr))
print(ridge.score(x_val, y_val))


# In[123]:


prediction=np.exp(ridge.predict(test))
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/submission_4.csv',index=False)


# ### can we remove some features?

# In[124]:


features_ridge={}
i = 0
for name in x_tr.columns:
    features_ridge[name]=ridge.coef_[i]
    i += 1
    
features_ridge_reduced={}
for name in features_ridge:
    if features_ridge[name] > 1e-07:
        features_ridge_reduced[name]=features_ridge[name]
        
reduced_features=[]
for i in features_ridge_reduced.keys():
    reduced_features.append(i)


# ### Refit the Ridge model with reduced features
# 112 remaining features: 'LotFrontage',
#  'LotArea',
#  'YearBuilt',
#  'YearRemodAdd',
#  'MasVnrArea',
#  'BsmtFinSF1',
#  'BsmtFinSF2',
#  'BedroomAbvGr',
#  'TotRmsAbvGrd',
#  'Fireplaces',
#  'GarageYrBlt',
#  'GarageCars',
#  'GarageArea',
#  'WoodDeckSF',
#  'EnclosedPorch',
#  '3SsnPorch',
#  'ScreenPorch',
#  'PoolArea',
#  'MiscVal',
#  'IsPool',
#  'IsGarage',
#  'TotalFullBath',
#  'TotalHalfBath',
#  'TotalSF',
#  'IsOpenPorch',
#  'MoSold_5',
#  'MoSold_6',
#  'MoSold_7',
#  'MoSold_8',
#  'MoSold_9',
#  'YrSold_2006',
#  'YrSold_2008',
#  'YrSold_2010',
#  'OverallQual_7',
#  'OverallQual_8',
#  'OverallQual_9',
#  'OverallQual_10',
#  'OverallCond_6',
#  'OverallCond_7',
#  'OverallCond_8',
#  'OverallCond_9',
#  'Exterior1st_MetalSd',
#  'Exterior1st_Other',
#  'Exterior1st_VinylSd',
#  'Condition1_Norm',
#  'LotShape_Reg',
#  'FireplaceQu_Fireplace',
#  'Electrical_Other',
#  'RoofMatl_Other',
#  'RoofStyle_Hip',
#  'RoofStyle_Other',
#  'Heating_Other',
#  'Foundation_Other',
#  'Foundation_PConc',
#  'SaleType_New',
#  'SaleType_Other',
#  'LandContour_Bnk',
#  'MSZoning_FV',
#  'MSZoning_RH',
#  'MSZoning_RL',
#  'Street_Pave',
#  'Alley_Pave',
#  'HouseStyle_1.5Fin',
#  'HouseStyle_2.5Unf',
#  'HouseStyle_2Story',
#  'BldgType_1Fam',
#  'BldgType_2fmCon',
#  'BldgType_Duplex',
#  'LandSlope_Mod',
#  'LotConfig_Corner',
#  'LotConfig_CulDSac',
#  'Neighborhood_Blmngtn',
#  'Neighborhood_BrkSide',
#  'Neighborhood_ClearCr',
#  'Neighborhood_CollgCr',
#  'Neighborhood_Crawfor',
#  'Neighborhood_NPkVill',
#  'Neighborhood_NoRidge',
#  'Neighborhood_NridgHt',
#  'Neighborhood_Somerst',
#  'Neighborhood_StoneBr',
#  'Neighborhood_Veenker',
#  'ExterCond_Ex',
#  'ExterCond_TA',
#  'ExterQual_Ex',
#  'ExterQual_Gd',
#  'GarageType_Attchd',
#  'GarageType_BuiltIn',
#  'PavedDrive_Y',
#  'KitchenQual_Ex',
#  'Fence_GdPrv',
#  'Fence_MnPrv',
#  'Fence_No Fence',
#  'MasVnrType_Stone',
#  'CentralAir_Y',
#  'SaleCondition_Alloca',
#  'SaleCondition_Normal',
#  'SaleCondition_Partial',
#  'HeatingQC_Ex',
#  'HeatingQC_Gd',
#  'BsmtFinType1_BLQ',
#  'BsmtFinType1_GLQ',
#  'BsmtExposure_Gd',
#  'BsmtExposure_Mn',
#  'BsmtCond_Gd',
#  'BsmtCond_TA',
#  'BsmtQual_Ex',
#  'GarageFinish_Fin',
#  'GarageFinish_RFn',
#  'GarageCond_Ex',
#  'GarageCond_Gd',
#  'GarageCond_TA'

# In[134]:


from sklearn.linear_model import Ridge
np.random.seed(1)
x_tr, x_val, y_tr, y_val = train_test_split(X, np.log(y), test_size = 0.2)
ridge_reduced = Ridge(normalize=True)
ridge_reduced.fit(x_tr[reduced_features], y_tr)
print(ridge_reduced.score(x_tr[reduced_features], y_tr))
print(ridge_reduced.score(x_val[reduced_features], y_val))


from sklearn.model_selection import GridSearchCV #Cross Validation
ridge_reduced = Ridge(normalize=True)
alphaRange = np.linspace(1e-3,1,100).tolist()
params = [{'alpha':alphaRange}]
gridSearch=GridSearchCV(estimator=ridge_reduced, param_grid=params)
gridSearch.fit(x_tr[reduced_features], y_tr)
print(gridSearch.best_params_)

ridge_reduced = Ridge(normalize=True,alpha=list(gridSearch.best_params_.values())[0])
ridge_reduced.fit(x_tr[reduced_features], y_tr)
print(ridge_reduced.score(x_tr[reduced_features], y_tr))
print(ridge_reduced.score(x_val[reduced_features], y_val))


# In[ ]:


# how do different coefficients shrink?
from sklearn.linear_model import Lasso
coefs  = []
scores = []
for alpha in alphas:
        lasso.set_params(normalize=True,alpha=alpha)
        lasso.fit(X, np.log(y))
        coefs.append(lasso.coef_)
        scores.append(lasso.score(X, np.log(y)))
coefs = pd.DataFrame(coefs, index = alphas, columns = X.columns)  

plt.rcParams['figure.figsize'] = (20,10)
for name in coefs.columns:
    plt.plot(coefs.index, coefs[name], label=name)
plt.legend(loc=4)   
plt.xlabel(r'hyperparameter $\lambda$')
plt.ylabel(r'slope values')


# ### Lasso 

# In[115]:


from sklearn.linear_model import Lasso
np.random.seed(1)
x_tr, x_val, y_tr, y_val = train_test_split(X, np.log(y), test_size = 0.2)
lasso = Lasso()
lasso.fit(x_tr, y_tr)
print(lasso.score(x_tr, y_tr))
print(lasso.score(x_val, y_val))


# In[111]:


from sklearn.model_selection import GridSearchCV #Cross Validation
alphaRange = np.linspace(1e-7,1e-4,100).tolist()
params = [{'alpha':alphaRange}]
gridSearch=GridSearchCV(estimator=lasso, param_grid=params)
gridSearch.fit(x_tr, y_tr)
print(gridSearch.best_params_)

lasso = Lasso(normalize=True, alpha=list(gridSearch.best_params_.values())[0])
lasso.fit(x_tr, y_tr)
print(lasso.score(x_tr, y_tr))
print(lasso.score(x_val, y_val))


# In[116]:


prediction=np.exp(lasso.predict(test))
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/submission_5.csv',index=False)


# ### Remove some features

# In[117]:


features_lasso={}
i = 0
for name in x_tr.columns:
    features_lasso[name]=lasso.coef_[i]
    i += 1
    
features_lasso_reduced={}
for name in features_lasso:
    if features_lasso[name] > 1e-07:
        features_lasso_reduced[name]=features_lasso[name]
        
reduced_features=[]
for i in features_lasso_reduced.keys():
    reduced_features.append(i)


# In[118]:


reduced_features


# ### Refit the lasso model after removing some features
# 10 remaining features: 'LotArea',
#  'YearBuilt',
#  'YearRemodAdd',
#  'MasVnrArea',
#  'BsmtFinSF1',
#  'GarageYrBlt',
#  'GarageArea',
#  'WoodDeckSF',
#  'MiscVal',
#  'TotalSF'

# In[119]:


from sklearn.linear_model import Lasso
np.random.seed(1)
x_tr, x_val, y_tr, y_val = train_test_split(X, np.log(y), test_size = 0.2)
lasso_reduced = Lasso()
lasso_reduced.fit(x_tr[reduced_features], y_tr)
print(lasso_reduced.score(x_tr[reduced_features], y_tr))
print(lasso_reduced.score(x_val[reduced_features], y_val))

from sklearn.model_selection import GridSearchCV #Cross Validation
alphaRange = np.linspace(1e-7,1e-4,100).tolist()
params = [{'alpha':alphaRange}]
gridSearch=GridSearchCV(estimator=lasso_reduced, param_grid=params)
gridSearch.fit(x_tr[reduced_features], y_tr)
print(gridSearch.best_params_)

lasso_reduced = Lasso(normalize=True, alpha=list(gridSearch.best_params_.values())[0])
lasso_reduced.fit(x_tr[reduced_features], y_tr)
print(lasso_reduced.score(x_tr[reduced_features], y_tr))
print(lasso_reduced.score(x_val[reduced_features], y_val))


# In[120]:


prediction=np.exp(lasso_reduced.predict(test[reduced_features]))

pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub.to_csv('../Submissions/submission_5.csv',index=False)

