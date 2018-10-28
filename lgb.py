import lightgbm as lgb
from sklearn import *
import pandas as pd
import numpy as np
from kaggle_util import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_squared_error

debug = 0
folds = 2 if debug else 5
nrows = 1000 if debug else None
#from top scoring kernels and blends - for testing only
#sub1 = pd.read_csv('../input/best-ensemble-score-made-available-0-68/SHAZ13_ENS_LEAKS.csv')
#sub2 = pd.read_csv('../input/best-ensemble-score-made-available-0-67/SHAZ13_ENS_LEAKS.csv')
#sub3 = pd.read_csv('../input/feature-scoring-vs-zeros/leaky_submission.csv')

#standard
train = pd.read_csv('../input/train.csv', nrows = nrows)
test = pd.read_csv('../input/test.csv', nrows = nrows)
print(train.shape, test.shape)

from PIL import Image, ImageDraw, ImageColor

col = [c for c in train.columns if c not in ['ID', 'target']]
xtrain = train[col].copy().values
target = train['target'].values

"""
im = Image.new('RGBA', xtrain.shape)
wh = ImageColor.getrgb('white')
re = ImageColor.getrgb('red')
gr = ImageColor.getrgb('green')
ga = ImageColor.getrgb('gray')

for x in range(xtrain.shape[0]):
    for y in range(xtrain.shape[1]):
        if xtrain[x][y] == 0:
            im.putpixel((x,y), wh)
        elif xtrain[x][y] == target[x]:
            im.putpixel((x,y), re)
        elif (np.abs(xtrain[x][y] - target[x]) / target[x]) < 0.05:
            im.putpixel((x,y), gr)
        else:
            im.putpixel((x,y), ga)
im.save('leak.bmp')
"""

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

#https://www.kaggle.com/johnfarrell/baseline-with-lag-select-fake-rows-dropped
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
    
from scipy.sparse import csr_matrix, vstack
train = train.replace(0, np.nan)
test = test.replace(0, np.nan)

test['target'] = 0.0

scores = []
y_label = np.log1p(train.target.values)
skf = StratifiedKFold(y_label, n_folds=folds)
X_train = train[col].values

for i, (train_split, val_split) in enumerate(skf):
    print('fold', i)
    x1 = X_train[train_split]
    y1 = y_label[train_split]
    x2 = X_train[val_split]
    y2 = y_label[val_split]

    params = {'learning_rate': 0.015, 
              'max_depth': 7, 
              'boosting': 'gbdt', 
              'objective': 'regression', 
              'metric': 'rmse', 
              'is_training_metric': True, 
              'num_leaves': 25,
              'lambda_l2': 1.0,
              'feature_fraction': 0.9, 
              'bagging_fraction': 0.8, 
              'bagging_freq': 5, 
              'seed':folds,
              'nthread': 5,
             }
    model = lgb.train(params, lgb.Dataset(x1, label=y1), 5000, lgb.Dataset(x2, label=y2), verbose_eval=200, early_stopping_rounds=100)
    test['target'] += np.expm1(model.predict(test[col], num_iteration=model.best_iteration))
    
    rmse = np.sqrt(mean_squared_error(y2, model.predict(x2)))
    scores.append(rmse)
    
test['target'] /= folds
#test[['ID', 'target']].to_csv('submission.csv', index=False)
save_result(test[['ID', 'target']], 
            '../result/lgb_{:.3f}.csv'.format(np.mean(scores)), 
            competition = 'santander-value-prediction-challenge', 
            send = False, 
            index = False)

"""
b1 = sub3.rename(columns={'target':'dp1'})
b2 = pd.read_csv('blend03.csv').rename(columns={'target':'dp2'})
b1 = pd.merge(b1, b2, how='left', on='ID')
b1['target'] = (b1['dp1'] * 0.6) + (b1['dp2'] * 0.4)
b1[['ID','target']].to_csv('blend04.csv', index=False)
"""