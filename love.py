import lightgbm as lgb
from sklearn import *
import pandas as pd
import numpy as np
from kaggle_util import *

#from top scoring kernels and blends - for testing only
sub3 = pd.read_csv('../result/scored_bayes_0.65.csv')

#standard
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)

col = [c for c in train.columns if c not in ['ID', 'target']]
xtrain = train[col].copy().values
target = train['target'].values

leak_col = []
for c in col:
    leak1 = np.sum((train[c]==train['target']).astype(int))
    leak2 = np.sum((((train[c] - train['target']) / train['target']) < 0.05).astype(int))
    if leak1 > 30 and leak2 > 3500:
        leak_col.append(c)
print(len(leak_col))

col = list(leak_col)
train = train[col +  ['ID', 'target']]
test = test[col +  ['ID']]

train["nz_mean"] = train[col].apply(lambda x: x[x!=0].mean(), axis=1)
train["nz_max"] = train[col].apply(lambda x: x[x!=0].max(), axis=1)
train["nz_min"] = train[col].apply(lambda x: x[x!=0].min(), axis=1)
train["ez"] = train[col].apply(lambda x: len(x[x==0]), axis=1)
train["mean"] = train[col].apply(lambda x: x.mean(), axis=1)
train["max"] = train[col].apply(lambda x: x.max(), axis=1)
train["min"] = train[col].apply(lambda x: x.min(), axis=1)

test["nz_mean"] = test[col].apply(lambda x: x[x!=0].mean(), axis=1)
test["nz_max"] = test[col].apply(lambda x: x[x!=0].max(), axis=1)
test["nz_min"] = test[col].apply(lambda x: x[x!=0].min(), axis=1)
test["ez"] = test[col].apply(lambda x: len(x[x==0]), axis=1)
test["mean"] = test[col].apply(lambda x: x.mean(), axis=1)
test["max"] = test[col].apply(lambda x: x.max(), axis=1)
test["min"] = test[col].apply(lambda x: x.min(), axis=1)
col += ['nz_mean', 'nz_max', 'nz_min', 'ez', 'mean', 'max', 'min']

for i in range(2, 100):
    train['index'+str(i)] = ((train.index + 2) % i == 0).astype(int)
    test['index'+str(i)] = ((test.index + 2) % i == 0).astype(int)
    col.append('index'+str(i))
    
test = pd.merge(test, sub3, how='left', on='ID',)

from scipy.sparse import csr_matrix, vstack
train = train.replace(0, np.nan)
test = test.replace(0, np.nan)
train = pd.concat((train, test), axis=0, ignore_index=True)
print('train.shape', train.shape)
test['target'] = 0.0

from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest MSE: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results_love.csv")
    
import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
"""
bayes_cv_tuner = BayesSearchCV(
    estimator = lgb.LGBMRegressor(objective='regression', 
                                  boosting_type='gbdt', 
                                  nthread= 5,
                                  verbose = -1
                                 ),
    search_spaces = {
        'learning_rate': (0.001, 0.1, 'log-uniform'),
        'num_leaves': (10, 200),      
        'max_depth': (3, 40), 
        'min_child_samples': (1, 50),
        'max_bin': (25, 500),
        'subsample_freq': (1, 5),
        'subsample':(0.1,0.95),
        'min_child_weight': (1e-4, 10),
        'reg_lambda': (1e-4, 200, 'log-uniform'),
        'reg_alpha': (1e-4, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-2, 500, 'log-uniform'),
        'n_estimators': (20, 100),
        'colsample_bytree':(0.5, 0.9),
    },    
    scoring = 'neg_mean_squared_log_error',
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 1,
    n_iter = 100,   
    verbose = 0,
    refit = True,
    random_state = 42
)

result = bayes_cv_tuner.fit(train[col], np.log1p(train.target.values), callback=status_print)
with open('best_params.txt', 'w+') as fo:
    fo.write(str(bayes_cv_tuner.best_params_))
"""    
    
folds = 5
for fold in range(folds):
    x1, x2, y1, y2 = model_selection.train_test_split(train[col], np.log1p(train.target.values), test_size=0.20, random_state=fold)
    params = {'colsample_bytree': 0.8974862424762783, 'learning_rate': 0.09980706205583974, 'max_bin': 29, 'max_depth': 40, 'min_child_samples': 44, 'min_child_weight': 9.219523888828594, 'n_estimators': 99, 'num_leaves': 29, 'reg_alpha': 0.0005494751938513909, 'reg_lambda': 87.21837263572203, 'scale_pos_weight': 383.5743832138119, 'subsample': 0.9139137737019314, 'subsample_freq': 3}
    params.update({
        'boosting': 'gbdt', 
        'objective': 'regression', 
        'metric': 'rmse', 
        'is_training_metric': True, 
        'seed':fold})
    model = lgb.train(params, lgb.Dataset(x1, label=y1), 3000, lgb.Dataset(x2, label=y2), verbose_eval=200, early_stopping_rounds=100)
    test['target'] += np.expm1(model.predict(test[col], num_iteration=model.best_iteration))
test['target'] /= folds

save_result(test[['ID', 'target']], 
            '../result/love_bayes.csv', 
            competition = 'santander-value-prediction-challenge', 
            send = True, 
            index = False)