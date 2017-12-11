#!/usr/bin/python
# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, model_selection
from sklearn.model_selection import cross_val_score

raw_data = pd.read_csv(r'./data/FEE.csv')
heads = raw_data.columns
# (0,1) transformation
scaler = MinMaxScaler(feature_range=(0, 1))
raw_data = pd.DataFrame(scaler.fit_transform(raw_data))
raw_data.columns = heads
# X,y
X = raw_data.drop(['OP_TIME', 'FEE'], axis=1)

poly_X = X
y = raw_data['FEE']
kf = KFold(n_splits=10)

kf.get_n_splits(poly_X)

print("start training Lasso model...")
# nested 10-fold cross-validation
scores = []
lasso_models = []
for train_index, test_index in kf.split(poly_X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    print("start training...")
    X_train, X_test = poly_X.ix[train_index], poly_X.ix[test_index]
    y_train, y_test = y.ix[train_index], y.ix[test_index]
    lassocv = linear_model.LassoCV(cv=10, max_iter=1500)
    lassocv.fit(X_train, y_train)
    print ("alpha is %s" % (lassocv.alpha_))
    lasso = linear_model.Lasso(alpha=lassocv.alpha_).fit(X_train, y_train)
    lasso_models.append(lasso)
    score = lasso.score(X_test, y_test)
    # print ('the score is %f' % (score))
    scores.append(score)

scores_ndarray = np.asarray(scores)
best_model = lasso_models[scores_ndarray.argmax()]
cv_result = model_selection.cross_val_score(best_model, poly_X, y, cv=kf, scoring='neg_mean_squared_error')
print ('the mean neg_mse_score for LassoRegression is %s' % (np.mean(np.asarray(cv_result))))

factors = np.asarray(best_model.coef_)
# intercept = best_model.intercept_
influence = factors/np.sum(factors)
def format_to_two_decimal_places(x):
    '''
    把浮点数格式化成两位小数
    :param x:
    :return:
    '''
    return '%.2f' % x

formatted_influence = map(format_to_two_decimal_places,influence)
X_heads = X.columns
print formatted_influence
#data = [BILL_FEE,PREPAY_FEE,PRESENT_FEE,INVALID_POST_BILL_FEE,SUBTRACT_FEE,OWEBACK_FEE]
# data = [300286570.1,42111148.77,-9533503.41,-13228092.72,-2130270.05,4011640.02]
# def income_explain(intercept,BILL_FEE,PREPAY_FEE,PRESENT_FEE,INVALID_POST_BILL_FEE,SUBTRACT_FEE,OWEBACK_FEE,factors):
# def fee_explain(intercept, factors, data):
#      # print ([w * x for w in factors,x in data])
#      total_
#     for i in range(len(factors)):
#         factors[i] * data[i]

# fee_explain(intercept,factors,data)
