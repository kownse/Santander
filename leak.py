import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from kaggle_util import *
from tqdm import tqdm

def add_leak(data, name, leak_path):
    leak = pd.read_csv(leak_path)
    data[name] = leak['compiled_leak'].values
    data['log_'+name] = np.log1p(leak['compiled_leak'].values)
    transact_cols = [f for f in leak.columns if f not in ["ID", "target","compiled_leak"]]
    leak["nonzero_mean"] = leak[transact_cols].apply(
            lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
        )
    data['log_nonzero_mean_'+name] = np.log1p(leak['nonzero_mean'])
    return data

def add_bulk_leak(data, leak_path):
    sort_leak = pd.read_csv(leak_path)
    #data = pd.concat([data, sort_leak], axis=1)
    cols = []
    for col in sort_leak.columns:
        if '_' not in col:
            
            log_name = 'log_' + col
            data[log_name] = np.log1p(sort_leak[col].values)
            cols.append(log_name)
    return data, cols
data = pd.read_csv('../input/train.csv')
target = np.log1p(data['target'])
data.drop(['ID', 'target'], axis=1, inplace=True)

#data = add_leak(data, 'leak', '../input/train_leak.csv')
#data = add_leak(data, 'leak6', '../input/train_leak_new6.csv')
#data = add_leak(data, 'leak16', '../input/train_leak_new16.csv')
#data = add_leak(data, 'leak22', '../input/train_leak_new22.csv')
data = add_leak(data, 'leak_tsne', '../input/train_leak_tsne.csv')
#data = add_leak(data, 'leak_tsne11', '../input/train_leak_tsne_11.3.csv')

leak_cols = [
    #'log_leak', 'log_nonzero_mean_leak',
    #'log_leak6', 'log_nonzero_mean_leak6',
    #'log_leak16', 'log_nonzero_mean_leak16',
    #'log_leak22', 'log_nonzero_mean_leak22',
    'log_leak_tsne', 'log_nonzero_mean_leak_tsne',
    #'log_leak_tsne11', 'log_nonzero_mean_leak_tsne11',
]

#data, cols = add_bulk_leak(data, '../input/bunk_leak.csv')
#leak_cols = cols[:25]
#print('leak cols:', leak_cols)
#exit()

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** .5
"""
reg = XGBRegressor(n_estimators=5000, nthread =8)
folds = KFold(5, True, 134257)
fold_idx = [(trn_, val_) for trn_, val_ in folds.split(data)]
scores = []

nb_values = data.nunique(dropna=False)
nb_zeros = (data == 0).astype(np.uint8).sum(axis=0)

features = [f for f in data.columns if f not in (['target', 'ID'] + leak_cols)]
for _f in tqdm(features):
    score = 0
    for trn_, val_ in fold_idx:
        reg.fit(
            data[leak_cols + [_f]].iloc[trn_], target.iloc[trn_],
            eval_set=[(data[leak_cols + [_f]].iloc[val_], target.iloc[val_])],
            eval_metric='rmse',
            early_stopping_rounds=100,
            verbose=2000
        )
        score += rmse(target.iloc[val_], reg.predict(data[leak_cols + [_f]].iloc[val_], ntree_limit=reg.best_ntree_limit)) / folds.n_splits
    scores.append((_f, score))
    
report = pd.DataFrame(scores, columns=['feature', 'rmse']).set_index('feature')
report['nb_zeros'] = nb_zeros
report['nunique'] = nb_values
report.sort_values(by='rmse', ascending=True, inplace=True)
report.to_csv('../input/report_clean.csv', index=True)
exit()
"""



from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb

folds = KFold(n_splits=5, shuffle=True, random_state=1)

report = pd.read_csv('../input/report.csv', index_col = 'feature')

good_features = report.loc[report['rmse'] <= 0.7955].index
print(good_features)

def calc_statistic(df, basefeatures):
    df.replace(0, np.nan, inplace=True)
    basedata = df[basefeatures].replace(0, np.nan)
    df['log_of_mean'] = np.log1p(basedata.mean(axis=1))
    df['mean_of_log'] = np.log1p(basedata).mean(axis=1)
    df['log_of_median'] = np.log1p(basedata.median(axis=1))
    df['nb_nans'] = df[basefeatures].isnull().sum(axis=1)
    df['the_sum'] = np.log1p(basedata).sum(axis=1)
    df['the_std'] = np.log1p(basedata).std(axis=1)
    df['the_kur'] = np.log1p(basedata).kurtosis(axis=1)
    return df

# Use all features for stats
basefeatures = [f for f in data if f not in (['ID', 'leak'] + leak_cols)]
data = calc_statistic(data, basefeatures)

# Only use good features, log leak and stats for training
features = good_features.tolist()
features = features + leak_cols + [
    'log_of_mean', 'mean_of_log', 'log_of_median', 'nb_nans', 
    'the_sum', 'the_std', 'the_kur']

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
        np.round(bayes_cv_tuner.best_score_, 8),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")
    
import math


#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

estimator = lgb.LGBMRegressor(objective='regression', 
                                  boosting_type='gbdt', 
                                  nthread= 4,
                                  verbose = -1
                                 )

search_spaces = {
        'learning_rate': (0.001, 0.1, 'log-uniform'),
        'num_leaves': (10, 200),      
        'max_depth': (3, 50), 
        'min_child_samples': (1, 100),
        'max_bin': (25, 900),
        'subsample_freq': (1, 10),
        'subsample':(0.5,1),
        'min_child_weight': (1e-4, 50),
        'reg_lambda': (1e-4, 500, 'log-uniform'),
        'reg_alpha': (1e-4, 10, 'log-uniform'),
        'scale_pos_weight': (1e-2, 600, 'log-uniform'),
        'n_estimators': (20, 500),
        'colsample_bytree':(0.4, 0.99),
    }


"""
estimator = XGBRegressor(n_estimators=5000, nthread =4)

search_spaces = {
        'learning_rate': (0.001, 0.1, 'log-uniform'),
        'max_depth': (3, 50), 
        'gamma': (0, 0.2),
        'min_child_weight': (1e-4, 50),
        'max_delta_step': (0, 0.5),
        'subsample':(0.5,1),
        'colsample_bytree':(0.4, 0.99),
        'colsample_bylevel':(0.1,1),
        'reg_lambda': (1e-4, 500, 'log-uniform'),
        'reg_alpha': (1e-4, 10, 'log-uniform'),
        'scale_pos_weight': (1e-2, 600, 'log-uniform'),
        'base_score':(0.1,0.9),
    }
"""


bayes_cv_tuner = BayesSearchCV(
    estimator = estimator,
    search_spaces = search_spaces, 
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


# Fit the model
result = bayes_cv_tuner.fit(data[features], target, callback=status_print)
with open('best_params.txt', 'w+') as fo:
    fo.write(str(bayes_cv_tuner.best_params_))
    
pred = bayes_cv_tuner.predict(data[features])
data['pred'] = pred
data['pred'].to_csv('../result/train_pre.csv')
data['combine'] = data.loc[data['leak_tsne'].isnan()
    
nrows = None
test = pd.read_csv('../input/test.csv', nrows = nrows)
#test = add_leak(test, 'leak', '../input/test_leak.csv')
#test = add_leak(test, 'leak6', '../input/test_leak_new6.csv')
#test = add_leak(test, 'leak16', '../input/test_leak_new16.csv')
#test = add_leak(test, 'leak22', '../input/test_leak_new22.csv')
#test = add_leak(test, 'leak_tsne', '../input/test_leak_tsne.csv')
test = add_leak(test, 'leak_tsne', '../input/test_leak_tsne.csv')
#test = add_leak(test, 'leak_tsne11', '../input/test_leak_tsne_11.3.csv')
#data, cols = add_bulk_leak(data, '../input/bunk_leak_test.csv')

test.replace(0, np.nan, inplace=True)
test = calc_statistic(test, basefeatures)
test['target'] = 0


pred = bayes_cv_tuner.predict(test[features])
test['target'] = np.expm1(pred)

save_result(test[['ID', 'target']], 
            '../result/bayes_leak_tsne_xgb.csv', 
            competition = 'santander-value-prediction-challenge', 
            send = False, 
            index = False)
exit()

"""
lgb_params = {
    'objective': 'regression',
    'num_leaves': 18,
    'subsample': 0.6143,
    'colsample_bytree': 0.6453,
    'min_split_gain': np.power(10, -2.5988),
    'reg_alpha': np.power(10, -2.2887),
    'reg_lambda': np.power(10, 1.7570),
    'min_child_weight': np.power(10, -0.1477),
    'verbose': 1,
    'seed': 3,
    'boosting_type': 'gbdt',
    'max_depth': -1,
    'learning_rate': 0.02,
    'metric': 'l2',
    'feature_fraction': 0.9, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 5,
    'nthread': 5,
}

lgb_params = {
    'objective': 'regression',
    'colsample_bytree': 0.8922912868445911, 'learning_rate': 0.09983658814703977, 'max_bin': 409, 'max_depth': 38, 'min_child_samples': 5, 'min_child_weight': 9.527639359465557, 'n_estimators': 83, 'num_leaves': 194, 'reg_alpha': 0.0008117455165837566, 'reg_lambda': 58.46156280543154, 'scale_pos_weight': 0.04068986891035624, 'subsample': 0.7577947527389557, 'subsample_freq': 3,
    'metric': 'l2',
    #'feature_fraction': 0.9, 
    'nthread': 5,
}

lgb_params = {'colsample_bytree': 0.7405906817659361, 'learning_rate': 0.09681620332282496, 'max_bin': 131, 'max_depth': 9, 'min_child_samples': 19, 'min_child_weight': 29.278503611768354, 'n_estimators': 69, 'num_leaves': 196, 'reg_alpha': 0.00026445638081503764, 'reg_lambda': 43.9319844743391, 'scale_pos_weight': 0.8034983909959849, 'subsample': 0.9888125423477032, 'subsample_freq': 10}


#lgb_params = bayes_cv_tuner.best_params_

lgb_params.update({
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
})

dtrain = lgb.Dataset(data=data[features], 
                     label=target, free_raw_data=False)
dtrain.construct()
oof_preds = np.zeros(data.shape[0])

scores = []
for trn_idx, val_idx in folds.split(data):
    clf = lgb.train(
        params=lgb_params,
        train_set=dtrain.subset(trn_idx),
        valid_sets=dtrain.subset(val_idx),
        num_boost_round=10000, 
        early_stopping_rounds=100,
        verbose_eval=0
    )

    
    oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
    test['target'] += clf.predict(test[features]) / folds.n_splits
    score = (mean_squared_error(target.iloc[val_idx], 
                             oof_preds[val_idx]) ** .5)
    scores.append(score)

data['predictions'] = oof_preds
print('OOF SCORE : %9.6f' 
      % (mean_squared_error(target, oof_preds) ** .5))
print('OOF SCORE with LEAK : %9.6f' 
      % (mean_squared_error(target, data['predictions']) ** .5))


#train['target']
test['target'] = np.expm1(test['target'])
#test.loc[test['leak'].notnull(), 'target'] = test.loc[test['leak'].notnull(), 'leak']
#test[['ID', 'target']].to_csv('leaky_submission.csv', index=False, float_format='%.2f')
print('ave score:', np.mean(scores))
save_result(test[['ID', 'target']], 
            '../result/bayes_kfold_old_{:.3f}.csv'.format(np.mean(scores)), 
            competition = 'santander-value-prediction-challenge', 
            send = False, 
            index = False)
"""