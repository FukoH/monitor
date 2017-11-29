#!/usr/bin/python
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model, model_selection


class FactorAnalyser(object):
    __factor_index_id = []
    __target = []

    def analyse(self, period, init):
        '''
        分析主方法
        :param period:  分析哪个期间
        :param init: 是否第一次初始化
        :return:
        '''
        data = self.__get_data_by_period(period)
        dic = self.__modeling(data)
        self.__save_result(dic, init)

    def __get_data_from_csv(self, path=r'./data/raw_data.csv'):
        '''
        从原始的excel把数据转成关系型
        :param path:
        :return:
        '''
        raw_data = pd.read_csv(path)
        raw_data['OP_TIME'].astype('int')
        model_data = raw_data[['OP_TIME', 'INDEX_ID', 'INDEX_VALUE']][
            raw_data['INDEX_ID'].isin(self.__class__.__factor_index_id)]
        output = raw_data[['OP_TIME', 'INDEX_VALUE']][raw_data['INDEX_ID'].isin(self.__class__.__target)]
        output.columns = ['OP_TIME', self.__class__.__target[0]]
        for column in self.__class__.__factor_index_id:
            short = raw_data[['OP_TIME', 'INDEX_ID', 'INDEX_VALUE']][raw_data['INDEX_ID'].isin([list(column)])]
            short.columns = ['OP_TIME', column]
            output.merge(short, how='outer', on='OP_TIME')
        return output

    def __get_data_by_period(self, period):
        '''

        :param period:  生成哪个期间的数据
        :return:
        '''
        output = self.__get_data_from_csv()
        series = output[output['OP_TIME'] <= period & output['OP_TIME'] >= 201303]
        series.interplorate()
        return series

    def __modeling(self, data):
        '''

        :param data:
        :return:  因素名称和影响因子的字典
        '''
        heads = data.columns
        # (0,1) transformation
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = pd.DataFrame(scaler.fit_transform(data))
        data.columns = heads
        # X,y
        poly_X = data.drop(['OP_TIME', self.__class__.__target[0]], axis=1)
        y = data[self.__class__.__target[0]]
        kf = TimeSeriesSplit(n_splits=3)
        kf.get_n_splits(poly_X)
        print("start trainning model...")
        # nested 3-fold TimeSeries cross-validation
        scores = []
        lasso_models = []
        for train_index, test_index in kf.split(poly_X):
            print("finding relatively better alpha...")
            X_train, X_test = poly_X.iloc[train_index], poly_X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            lassocv = linear_model.LassoCV(cv=10, max_iter=1500)
            lassocv.fit(X_train, y_train)
            lasso = linear_model.Lasso(alpha=lassocv.alpha_).fit(X_train, y_train)
            lasso_models.append(lasso)
            score = lasso.score(X_test, y_test)
            scores.append(score)
        scores_ndarray = np.asarray(scores)
        best_model = lasso_models[scores_ndarray.argmax()]
        cv_result = model_selection.cross_val_score(best_model, poly_X, y, cv=kf, scoring='neg_mean_squared_error')
        print ('the mean neg_mse_score for LassoRegression is %s' % (np.mean(np.asarray(cv_result))))
        # 得到系数的list
        # factors = np.square(np.asarray(best_model.coef_))
        factors = np.abs(np.asarray(best_model.coef_))
        # 得到影响因子
        influence = factors / np.sum(factors)

        # 格式化一下小数,输出两位小数
        formatted_influence = map(lambda x: '%.2f' % x, influence)
        named_scores = zip(poly_X.columns, formatted_influence)
        # sorted_named_scores = sorted(named_scores, key=lambda influence: influence[1], reverse=True)
        return dict(named_scores)

    def __save_result(self, dic, init):
        raise NotImplementedError


class BillUserAnalyser(FactorAnalyser):
    pass


class NetBillUserAnalyser(FactorAnalyser):
    pass


class FeeAnalyser(FactorAnalyser):
    def __get_data_by_period(self, period):
        raise NotImplementedError

    def __modeling(self, data):
        raise NotImplementedError

    def __save_result(self, init):
        raise NotImplementedError


class FactorAnalyserFactory(object):
    @classmethod
    def get_analyser(cls, index_id):
        if index_id == 'ZB1001001':
            return BillUserAnalyser()
        elif index_id == 'ZB1001002':
            return NetBillUserAnalyser()
        elif index_id == 'ZB1001003':
            return FeeAnalyser()
