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

data = pd.read_csv('../input/train_clean.csv')
target = np.log1p(data['target'])
data.drop(['ID', 'target'], axis=1, inplace=True)

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

data = add_leak(data, 'leak', '../input/train_leak_clean.csv')
data = add_leak(data, 'leak6', '../input/train_leak_clean6.csv')
data = add_leak(data, 'leak16', '../input/train_leak_clean16.csv')
data = add_leak(data, 'leak22', '../input/train_leak_clean22.csv')

leak_cols = [
    'log_leak', 'log_nonzero_mean_leak',
    'log_leak6', 'log_nonzero_mean_leak6',
    'log_leak16', 'log_nonzero_mean_leak16',
    'log_leak22', 'log_nonzero_mean_leak22',
]

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

report = pd.read_csv('../input/report_clean.csv', index_col = 'feature')

good_features = report.loc[report['rmse'] <= 0.645].index
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

dtrain = lgb.Dataset(data=data[features], 
                     label=target, free_raw_data=False)

import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

N_HYPEROPT_PROBES = 500

HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest
def get_lgb_params(space):
    lgb_params = dict()
    lgb_params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
    lgb_params['application'] = 'regression'
    lgb_params['metric'] = 'l2_root'
    lgb_params['learning_rate'] = space['learning_rate']
    lgb_params['num_leaves'] = int(space['num_leaves'])
    lgb_params['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    lgb_params['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    lgb_params['max_depth'] = -1
    lgb_params['lambda_l1'] = space['lambda_l1'] if 'lambda_l1' in space else 0.0
    lgb_params['lambda_l2'] = space['lambda_l2'] if 'lambda_l2' in space else 0.0
    lgb_params['max_bin'] = int(space['max_bin']) if 'max_bin' in space else 256
    lgb_params['feature_fraction'] = space['feature_fraction']
    lgb_params['bagging_fraction'] = space['bagging_fraction']
    lgb_params['bagging_freq'] = int(space['bagging_freq']) if 'bagging_freq' in space else 1

    return lgb_params

obj_call_count = 0
cur_best_loss = np.inf


def objective(space):
    global obj_call_count, cur_best_loss
    obj_call_count += 1
    lgb_params = get_lgb_params(space)

    sorted_params = sorted(space.items(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])

    scores = []
    for trn_idx, val_idx in folds.split(data):
        model = lightgbm.train(lgb_params,
                               train_set=dtrain.subset(trn_idx),
                               valid_sets=dtrain.subset(val_idx),
                               num_boost_round=10000,
                               early_stopping_rounds=100,
                               verbose_eval=False,
                               )
        predict = model.predict(dtrain.data.iloc[val_idx], num_iteration=model.best_iteration)
        score = (mean_squared_error(target.iloc[val_idx], predict) ** .5)
        scores.append(score)
        

    test_loss = np.mean(scores)
    if test_loss<cur_best_loss:
        cur_best_loss = test_loss
        print('NEW BEST LOSS={}'.format(cur_best_loss))

    return{'loss':test_loss, 'status': STATUS_OK }

space ={
        'num_leaves': hp.quniform ('num_leaves', 10, 200, 1),
        'min_data_in_leaf':  hp.quniform ('min_data_in_leaf', 10, 200, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'max_bin': hp.quniform ('max_bin', 64, 512, 1),
        'bagging_freq': hp.quniform ('bagging_freq', 1, 5, 1),
        'lambda_l1': hp.uniform('lambda_l1', 0, 10 ),
        'lambda_l2': hp.uniform('lambda_l2', 0, 10 ),
       }

trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=0)

exit()


nrows = None
test = pd.read_csv('../input/test.csv', nrows = nrows)
test = add_leak(test, 'leak', '../input/test_leak_clean.csv')
test = add_leak(test, 'leak6', '../input/test_leak_clean6.csv')
test = add_leak(test, 'leak16', '../input/test_leak_clean16.csv')
test = add_leak(test, 'leak22', '../input/test_leak_clean22.csv')


test.replace(0, np.nan, inplace=True)
test = calc_statistic(test, basefeatures)
test['target'] = 0

pred = bayes_cv_tuner.predict(test[features])
test['target'] = np.expm1(pred)
#test[['ID', 'target']].to_csv('my_submission.csv', index=False, float_format='%.2f')

save_result(test[['ID', 'target']], 
            '../result/bayes_leak_all.csv', 
            competition = 'santander-value-prediction-challenge', 
            send = False, 
            index = False)

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
"""
"""
lgb_params = {
    'objective': 'regression',
    'colsample_bytree': 0.8922912868445911, 'learning_rate': 0.09983658814703977, 'max_bin': 409, 'max_depth': 38, 'min_child_samples': 5, 'min_child_weight': 9.527639359465557, 'n_estimators': 83, 'num_leaves': 194, 'reg_alpha': 0.0008117455165837566, 'reg_lambda': 58.46156280543154, 'scale_pos_weight': 0.04068986891035624, 'subsample': 0.7577947527389557, 'subsample_freq': 3,
    'metric': 'l2',
    #'feature_fraction': 0.9, 
    'nthread': 5,
}

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
    #test['target'] += clf.predict(test[features]) / folds.n_splits
    score = (mean_squared_error(target.iloc[val_idx], 
                             oof_preds[val_idx]) ** .5)
    scores.append(score)

data['predictions'] = oof_preds
data.loc[data['leak'].notnull(), 'predictions'] = np.log1p(data.loc[data['leak'].notnull(), 'leak'])
print('OOF SCORE : %9.6f' 
      % (mean_squared_error(target, oof_preds) ** .5))
print('OOF SCORE with LEAK : %9.6f' 
      % (mean_squared_error(target, data['predictions']) ** .5))

df_oof = pd.DataFrame(oof_preds)
df_oof.to_csv('train_result.csv', index = False)
exit()

test['target'] = np.expm1(test['target'])
test.loc[test['leak'].notnull(), 'target'] = test.loc[test['leak'].notnull(), 'leak']
#test[['ID', 'target']].to_csv('leaky_submission.csv', index=False, float_format='%.2f')
print('ave score:', np.mean(scores))
save_result(test[['ID', 'target']], 
            '../result/bayes_{:.3f}.csv'.format(np.mean(scores)), 
            competition = 'santander-value-prediction-challenge', 
            send = False, 
            index = False)
"""