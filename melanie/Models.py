#!/usr/bin/env python
# coding: utf-8

# In[221]:


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import re
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')


# In[222]:


# Load libraries
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor as rfr,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


# In[223]:


from __future__ import print_function  # Python 2 and 3
from IPython.display import display
pd.options.display.max_columns = 100
pd.options.display.max_rows = 500


# In[224]:


train = pd.read_csv('../data/train_processed.csv')
test_raw = pd.read_csv('../data/test.csv')
test = pd.read_csv('../data/test_processed.csv')


# In[227]:


#one time fix --- resolved in preprocess.py
# train=train.drop(columns=['Unnamed: 0'])
# test=test.drop(columns=['Unnamed: 0'])


# In[228]:


y = train['SalePrice'] # target variable
yt = stats.boxcox(y, lmbda = 0.3) # Boxcox
X = train.loc[:, train.columns!='SalePrice']
X = X.loc[:, X.columns!='HouseStyle_2.5Fin'] #HouseStyle_2.5Fin in training.csv but not in test.csv


# In[229]:


print(test.shape)
print(X.shape)


# ## Skewness (adjust Skewness of numeric columns)
# #### 
# Unnamed: 0                    int64
# MSSubClass                    int64
# LotFrontage                 float64
# LotArea                       int64
# YearBuilt                     int64
# YearRemodAdd                  int64
# MasVnrArea                  float64
# BsmtFinSF1                  float64
# BsmtFinSF2                  float64
# BedroomAbvGr                  int64
# KitchenAbvGr                  int64
# TotRmsAbvGrd                  int64
# Fireplaces                    int64
# GarageYrBlt                 float64
# GarageCars                  float64
# GarageArea                  float64
# WoodDeckSF                    int64
# EnclosedPorch                 int64
# 3SsnPorch                     int64
# ScreenPorch                   int64
# PoolArea                      int64
# MiscVal                       int64
# IsPool                        int64 - not numerical
# IsGarage                      int64 - not numerical
# TotalFullBath               float64
# TotalHalfBath               float64
# TotalSF                     float64

# In[230]:


numeric=X.loc[:, X.columns != 'IsPool']
numeric = numeric.loc[:, numeric.columns !='IsGarage']
numerical = numeric.dtypes[:27].index.to_list()


# In[231]:


skewed = X[numerical].apply(lambda x: x.skew()).sort_values()
skewdf = pd.DataFrame({'Skew': skewed})
skewdf.head(3)
skewdf = skewdf[(skewdf)>0.75]
from scipy.special import boxcox1p
skewed = skewdf.index
lam = 0.15
for feat in skewed:
    X[feat] = boxcox1p(X[feat], lam)
    test[feat] = boxcox1p(test[feat], lam)
# newskewed = X[numerical].apply(lambda x: x.skew()).sort_values()


# ## Evaluate Models

# In[233]:


### This function returns R2 for test and validation sets
def get_r2(model): 
    R2_train=model.score(x_tr, y_tr)
    R2_val=model.score(x_val,y_val)
    return(R2_train, R2_val)


### This function returns RMSE for 5 fold Cross Validation tests.
def rmsle_cv(model, boxcox=True):
    if boxcox == True:
        y_ = yt
    else:
        y_ = np.log(y)
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=1).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# ## Model 1: OLM

# ### OLM - Log(y)

# In[247]:


x_tr, x_val, y_tr, y_val = train_test_split(X, np.log(y), test_size=0.2, random_state=1)
olm = LinearRegression()
olm.fit(x_tr, y_tr)

print(f"R2 for train and test are: {get_r2(olm)}")
olm_score=rmsle_cv(olm, boxcox=False)
print("Olm RMSE: {:.4f} ({:.4f})".format(olm_score.mean(), olm_score.std()))

prediction=np.exp(olm.predict(test))
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub.to_csv('../Submissions/submission_olm_logY.csv',index=False)


# ### OLM - BoxCox on y

# In[249]:


x_tr, x_val, y_tr, y_val = train_test_split(X, yt, test_size=0.2, random_state=1)
olm = LinearRegression()
olm.fit(x_tr, y_tr)
print(f"R2 for train and test are: {get_r2(olm)}")

olm_score=rmsle_cv(olm)
print("Olm RMSE: {:.4f} ({:.4f})".format(olm_score.mean(), olm_score.std()))

prediction=olm.predict(test)
prediction=inv_boxcox(prediction,0.3)
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/submission_olm_boxcox.csv',index=False)


# #### BoxCox seems to perform slightly better than log transformation of SalePrice. Therefore, we will train the rest of the models based on BoxCox transformation
# 

# ## How do the coefficients shrink?

# In[236]:


# how do different coefficients shrink?
from sklearn.linear_model import Lasso
lasso = Lasso(normalize=True)
alphas = np.linspace(1e-9,0.5,100)
lasso.set_params(normalize=True)
coefs  = []
scores = []
for alpha in alphas:
        lasso.set_params(alpha=alpha)
        lasso.fit(X, yt)
        coefs.append(lasso.coef_)
        scores.append(lasso.score(X, yt))
coefs = pd.DataFrame(coefs, index = alphas, columns = X.columns)  

plt.rcParams['figure.figsize'] = (20,10)
for name in coefs.columns:
    plt.plot(coefs.index, coefs[name], label=name)
plt.legend(loc=4)   
plt.xlabel(r'hyperparameter $\lambda$')
plt.ylabel(r'slope values')


# ## Model 2: Ridge 

# In[238]:


from sklearn.linear_model import Ridge
x_tr, x_val, y_tr, y_val = train_test_split(X, yt, test_size = 0.2, random_state=1)
ridge = make_pipeline(RobustScaler(), Ridge(normalize=True))
ridge.fit(x_tr, y_tr)
print(f"R2 for train and test are: {get_r2(ridge)}")

from sklearn.model_selection import GridSearchCV #Cross Validation
ridge = Ridge(normalize=True)
alphaRange = np.linspace(1e-3,1,100).tolist()
params = [{'alpha':alphaRange}]
gridSearch=GridSearchCV(estimator=ridge, param_grid=params)
gridSearch.fit(x_tr, y_tr)

print(f"Ridge Best Params: {gridSearch.best_params_}")
ridge = gridSearch.best_estimator_
print(f"[Ridge best params] R2 for train and test are: {get_r2(ridge)}")

ridge_score=rmsle_cv(ridge)
print("[Ridge best params] RMSE: {:.4f} ({:.4f})".format(ridge_score.mean(), ridge_score.std()))


# In[250]:


prediction=ridge.predict(test)
prediction=inv_boxcox(prediction, 0.3)
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/submission_ridge.csv',index=False)


# ## Model 3: Lasso 

# In[253]:


from sklearn.linear_model import Lasso
np.random.seed(1)
x_tr, x_val, y_tr, y_val = train_test_split(X, yt, test_size = 0.2)
lasso = Lasso(normalize=True)

from sklearn.model_selection import GridSearchCV #Cross Validation
alphaRange = np.linspace(1e-3,1e3,1000).tolist()
params = {'alpha':alphaRange}
gridSearch=GridSearchCV(estimator=lasso, param_grid=params)
gridSearch.fit(x_tr, y_tr)
print(f"Lasso Best Params: {gridSearch.best_params_}") 

lasso_best = gridSearch.best_estimator_
lasso_best.fit(x_tr, y_tr)
print(f"[Lasso best params] R2 for train and test are: {get_r2(lasso_best)}")

lasso_score=rmsle_cv(lasso_best)
print("[Lasso best params] RMSE: {:.4f} ({:.4f})".format(lasso_score.mean(), lasso_score.std()))


# In[254]:


prediction=lasso_best.predict(test)
prediction=inv_boxcox(prediction, 0.3)
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/submission_lasso.csv',index=False)


# ### Lasso is performing better than Ridge
# ### Below are surviving features after Lasso

# In[255]:


features_lasso={}
i = 0
for name in x_tr.columns:
    features_lasso[name]=lasso_best.coef_[i]
    i += 1
    
features_lasso_reduced={}
for name in features_lasso:
    if features_lasso[name] > 1e-10:
        features_lasso_reduced[name]=features_lasso[name]
        
reduced_features=[]
for i in features_lasso_reduced.keys():
    reduced_features.append(i)
print(len(reduced_features))
print(reduced_features)


# ##### Post Lasso Features: (78 Survived) ... See above

# ### Elastic Net

# In[257]:


from sklearn.linear_model import ElasticNet
x_tr, x_val, y_tr, y_val = train_test_split(X, yt, test_size = 0.2, random_state=1)
net = ElasticNet()
net.fit(x_tr, y_tr)
print(f"R2 for train and test are: {get_r2(net)}")

from sklearn.model_selection import GridSearchCV #Cross Validation
alphaRange=np.linspace(1e-5,1e-2,60).tolist()
l1_ratio_range=np.linspace(0,1,30).tolist()
params = {'alpha':alphaRange, 'l1_ratio':l1_ratio_range}
gridSearch=GridSearchCV(estimator=net, param_grid=params)
gridSearch.fit(x_tr, y_tr)
print(gridSearch.best_params_)

net_best = gridSearch.best_estimator_
print(f"[Net best params] R2 for train and test are: {get_r2(net_best)}")


# In[69]:


net_score=rmsle_cv(net_best)
print("Net RMSE: {:.4f} ({:.4f})".format(net_score.mean(), net_score.std()))


# In[186]:


prediction=net_best.predict(test)
prediction=inv_boxcox(prediction, 0.3)
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/submission_net.csv',index=False)


# In[ ]:





# ## CatBoost

# In[155]:


from catboost import CatBoostRegressor
catB = CatBoostRegressor(iterations=1000)
score = rmsle_cv(catB)
print("CatBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
catB.fit(x_tr,y_tr)
print(f"R2 for Train, Test are: {get_r2(catB)}")


# ## Gradient Boosting
# #### fit a model on error every time

# In[145]:


gb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

score = rmsle_cv(gb)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
gb.fit(x_tr, y_tr)
print(f"R2 for Train, Test are: {get_r2(gb)}")


# #### Find best Gradient Boosting params

# In[146]:


x_tr, x_val, y_tr, y_val = train_test_split(X, yt, test_size = 0.2, random_state=1)

params = {"n_estimators":[3000,6000], "learning_rate":[0.3,0.6], "max_depth":[3,5], "min_samples_leaf":[10,20]}

from sklearn.model_selection import GridSearchCV #Cross Validation
gridSearch=GridSearchCV(estimator=gb, param_grid=params)
gridSearch.fit(x_tr, y_tr) # boxcox of y
print(gridSearch.best_params_)
# {'learning_rate': 0.3, 'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 3000}
gb_best = gridSearch.best_estimator_

print(f"R2 for Train, Test are: {get_r2(gb_best)}")

score = rmsle_cv(gb_best)
print("Gradient Boosting Best Params score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[187]:


prediction=gb_best.predict(test)
prediction=inv_boxcox(prediction, 0.3)
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/submission_GBoost.csv',index=False)


# ### XGBoost

# In[ ]:


import xgboost as xgb
xgboost = xgb.XGBRegressor()
score = rmsle_cv(xgboost)
#print("XG Boost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
xgboost.fit(x_tr, y_tr)
print(f"R2 for train, test are: {get_r2(xgboost)}")


# In[156]:


import xgboost as xgb
xgboost = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                       learning_rate=0.05, max_depth=3, 
                       min_child_weight=1.7817, n_estimators=2200,
                       reg_alpha=0.4640, reg_lambda=0.8571,
                       subsample=0.5213, silent=1,
                       random_state =5, nthread = -1)
score = rmsle_cv(xgboost)
print("XG Boost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
xgboost.fit(x_tr, y_tr)
print(f"R2 for train, test are: {get_r2(xgboost)}")


# In[84]:


# xgb.plot_importance(xgboost)
# plt.rcParams['figure.figsize'] = [200, 200]
# plt.show()


# ## LightGBM

# In[129]:


import lightgbm as lgb
lgb_ = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score_lgbm = rmsle_cv(lgb_)
print("LightGMB RMSE: {:.4f} ({:.4f})\n".format(score_lgbm.mean(), score_lgbm.std()))
lgb_.fit(x_tr, y_tr)
print(f"R2 for train, test are: {get_r2(lgb_)}")


# ### Try different hyper parameters
#  - max_bin: the maximum numbers of bins that feature values are bucketed in. A smaller max_bin reduces overfitting.
#  - min_child_weight: the minimum sum hessian for a leaf. In conjuction with min_child_samples, larger values reduce overfitting.
#  - bagging_fraction and bagging_freq: enables bagging (subsampling) of the training data. Both values need to be set for bagging to be used. The frequency controls how often (iteration) bagging is used. Smaller fractions and frequencies reduce overfitting.
#  - feature_fraction: controls the subsampling of features used for training (as opposed to subsampling the actual training data in the case of bagging). Smaller fractions reduce overfitting.

# In[121]:


import lightgbm as lgb
lgb_ = lgb.LGBMRegressor(objective='regression')
score_lgbm = rmsle_cv(lgb_)
print("LightGMB RMSE: {:.4f} ({:.4f})\n".format(score_lgbm.mean(), score_lgbm.std()))
lgb_.fit(x_tr, y_tr)
print(f"R2 for train, test are: {get_r2(lgb_)}")

# fine-tuning params
params = {
    'num_leaves': [2,5,10],
    'min_child_samples': [30, 45],
    'learning_rate': [0.03, 0.05, 0.07],
    "max_bin": [30,40,50],
    'feature_fraction': [0.2, 0.4, 0.6, 0.8]
    "n_estimators"=[500, 750, 1000],
    "bagging_fraction" = 0.8,
    bagging_freq = 5, 
    feature_fraction = 0.2319,
    feature_fraction_seed=9, 
    bagging_seed=9,
    min_data_in_leaf =10 
}


# ## Simple Averaging

# In[175]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[178]:


simple_stack_model = AveragingModels(models = (ridge, lasso_best, net_best, catB, gb_best, xgboost))

score = rmsle_cv(simple_stack_model)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Averaged base models score: 0.0591 (0.0075)


# In[188]:


simple_stack_model.fit(X, yt)
prediction=simple_stack_model.predict(test)
prediction=inv_boxcox(prediction, 0.3)
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub.to_csv('../Submissions/submission_simpleStacked.csv',index=False)


# ## Stacked (meta_model=lasso)

# StackingModels Class Source: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

# In[ ]:




