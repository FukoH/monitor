#!/usr/bin/python
# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, model_selection

raw_data = pd.read_csv(r'../data/BILL_USER.csv')
heads = raw_data.columns
# (0,1) transformation
scaler = MinMaxScaler(feature_range=(0, 1))
raw_data = pd.DataFrame(scaler.fit_transform(raw_data))
raw_data.columns = heads
# X,y
X = raw_data.drop(['OP_TIME', 'BILL_USER'], axis=1)
# poly = PolynomialFeatures(degree=2)

# poly_X = pd.DataFrame(poly.fit_transform(X))
# print(poly.get_feature_names(X.columns))
poly_X = X
y = raw_data['BILL_USER']
# kf = KFold(n_splits=10)
kf = TimeSeriesSplit(n_splits=3)

kf.get_n_splits(poly_X)
regressions = ['Lasso', 'Ridge', 'GradientBoostingRegression']

print("start trainning %s model..." % (regressions[0]))
# nested 10-fold cross-validation
scores = []
lasso_models = []
mean_performance = []
for train_index, test_index in kf.split(poly_X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    print("start training...")
    X_train, X_test = poly_X.iloc[train_index], poly_X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    lassocv = linear_model.LassoCV(cv=10, max_iter=1500)
    lassocv.fit(X_train, y_train)
    print ("alpha is %s" % (lassocv.alpha_))
    lasso = linear_model.Lasso(alpha=lassocv.alpha_).fit(X_train, y_train)
    lasso_models.append(lasso)
    score = lasso.score(X_test, y_test)
    scores.append(score)
scores_ndarray = np.asarray(scores)
best_model = lasso_models[scores_ndarray.argmax()]
cv_result = model_selection.cross_val_score(best_model, poly_X, y, cv=kf, scoring='neg_mean_squared_error')
print ('the mean neg_mse_score for LassoRegression is %s' % (np.mean(np.asarray(cv_result))))
mean_performance.append(np.mean(np.asarray(cv_result)))

# ===========================
# ridge_scores = []
# ridge_models = []
# for train_index, test_index in kf.split(poly_X):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     print("start training...")
#     X_train, X_test = poly_X.iloc[train_index], poly_X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#     lassocv = linear_model.RidgeCV(cv=10, alphas=[0.01, 0.03, 0.1, 0.3, 1, 3])
#     lassocv.fit(X_train, y_train)
#     print ("alpha is %s" % (lassocv.alpha_))
#     lasso = linear_model.Ridge(alpha=lassocv.alpha_).fit(X_train, y_train)
#     ridge_models.append(lasso)
#     score = lasso.score(X_test, y_test)
#     ridge_scores.append(score)
#
# ridge_scores_ndarray = np.asarray(ridge_scores)
# best_model = ridge_models[ridge_scores_ndarray.argmax()]
# cv_result = model_selection.cross_val_score(best_model, poly_X, y, cv=kf, scoring='neg_mean_squared_error')
# print ('the mean neg_mse_score for RidgeRegression is %s' % (np.mean(np.asarray(cv_result))))
# mean_performance.append(np.mean(np.asarray(cv_result)))
# # ========================
# params = {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2,
#           'learning_rate': 0.01, 'loss': 'ls'}
# clf_gbr = GradientBoostingRegressor(**params)
# cv_result = model_selection.cross_val_score(clf_gbr, poly_X, y, cv=kf, scoring='neg_mean_squared_error')
# print ('the mean neg_mse_score for GRB is %s' % (np.mean(np.asarray(cv_result))))
# mean_performance.append(np.mean(np.asarray(cv_result)))

# 得到系数的list
# factors = np.square(np.asarray(best_model.coef_))
factors = np.abs(np.asarray(best_model.coef_))

#得到影响因子
influence = factors/np.sum(factors)

def format_to_two_decimal_places(x):
    '''
    把浮点数格式化成两位小数
    :param x:
    :return:
    '''
    return '%.2f' % x
#格式化一下小数,输出两位小数
formatted_influence = map(format_to_two_decimal_places,influence)
named_scores = zip(X.columns, influence)
sorted_named_scores = sorted(named_scores, key=lambda influence: influence[1], reverse=True)
for (name,factor) in sorted_named_scores:
    print('%s , %.2f ' %(name,factor))

import seaborn as sns
sns.barplot(x=influence, y=X.columns,order=[s[0] for s in sorted_named_scores],orient='h')
plt.show()