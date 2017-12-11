#!/usr/bin/python
# -*- coding: utf-8 -*

import numpy as np
import pandas as pd


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, model_selection

raw_data = pd.read_csv(r'./data/FEE.csv')
heads = raw_data.columns
# (0,1) transformation
scaler = MinMaxScaler(feature_range=(0, 1))
# raw_data = pd.DataFrame(scaler.fit_transform(raw_data))
raw_data.columns = heads
# X,y
X = raw_data.drop(['OP_TIME', 'FEE'], axis=1)
# poly = PolynomialFeatures(degree=2)

# poly_X = pd.DataFrame(poly.fit_transform(X))
# print(poly.get_feature_names(X.columns))
poly_X = X
y = raw_data['FEE']
kf = KFold(n_splits=10)

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
    y_train, y_test = y.ix[train_index], y.ix[test_index]
    lassocv = linear_model.LassoCV(cv=10, max_iter=1500)
    lassocv.fit(X_train, y_train)
    print ("alpha is %s" % (lassocv.alpha_))
    lasso = linear_model.Lasso(alpha=lassocv.alpha_).fit(X_train, y_train)
    lasso_models.append(lasso)
    score = lasso.score(X_test, y_test)
    # print ('the score is %f' % (score))
    scores.append(score)
    # print('true value of test is')
    # print (y_test)
    # print('predict value of test is')
    # print (lasso.predict(X_test))
    # print('parameters is' )
    # print (lasso.coef_)
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
#     X_train, X_test = poly_X.ix[train_index], poly_X.ix[test_index]
#     y_train, y_test = y.ix[train_index], y.ix[test_index]
#     lassocv = linear_model.RidgeCV(cv=10, alphas=[0.01, 0.03, 0.1, 0.3, 1, 3])
#     lassocv.fit(X_train, y_train)
#     print ("alpha is %s" % (lassocv.alpha_))
#     lasso = linear_model.Ridge(alpha=lassocv.alpha_).fit(X_train, y_train)
#     ridge_models.append(lasso)
#     score = lasso.score(X_test, y_test)
#     # print ('the score is %f' % (score))
#     ridge_scores.append(score)
    # print('true value of test is')
    # print (y_test)
    # print('predict value of test is')
    # print (lasso.predict(X_test))
    # print('parameters is' )
    # print (lasso.coef_)
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



# avg = sum(scores) / len(scores)
# print("avg is %f" % (avg))


# cross_val_score(lasso,)
# lasso.score()
# print(lasso.mse_path_)
# print('==============')
# print(lasso.alphas_)
# alphas.append(lasso.alpha_)

# lasso = linear_model.Lasso(alpha=best_alpha, max_iter=1500)
# lasso.fit
# print scores
# mean_score = np.mean(np.asarray(scores))
# print ('mean_score is %f'% (mean_score))
# print(lasso.predict(poly_X.ix[test_index]))
