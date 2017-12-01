#!/usr/bin/python
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model, model_selection

from po.PO import Relation
from DB.DBConnector import DBConnector


class FactorAnalyser(object):
    __target = []
    __factor_index_id = []
    __level_two_factor_id = {}  # {'index1':[index1-a,index1-b],'index2':[index2-a,index2-b]}
    connector = DBConnector()

    def analyse(self, period, init):
        '''
        分析主方法
        :param period:  分析哪个期间
        :param init: 是否第一次初始化
        :return:
        '''
        # load data
        data = self.__get_data_by_period(period)
        # predict
        self.__predict(data,init)
        # calculate factor
        dic = self.__modeling(data)
        list = self.__save_result(dic, period, init)
        list_parent, list_root = self.__save_second_level(period, init)
        list.extend(list_parent).extend(list_root)
        self.__class__.connector.add_data(list)

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

    def __predict(self,data,init):
        self.__predict_by_model(data,init)
        self.__add_to_database(init)


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

    def __save_result(self, dic, period, init):
        list = []
        for key, value in dic.iteritems():
            r = Relation()
            r.index_pk_id = key
            r.influence_factor = value
            r.parent_pk_id = self.__target[0]
            r.op_time = period
            r.distance = 1
            r.is_leaf = 0
            list.append(r)
        return list

    def __save_second_level(self, period, init):
        list_parent = []  # 和父节点
        list_root_second = []  # 和根节点
        for key, value in self.__level_two_factor_id.iteritems():
            for v in value:
                r = Relation()
                r.index_pk_id = v
                r.influence_factor = None
                r.parent_pk_id = key
                r.op_time = period
                r.distance = 1
                r.is_leaf = 1
                list_parent.append(r)

                root = Relation()
                root.index_pk_id = v
                root.influence_factor = None
                root.parent_pk_id = self.__target[0]
                root.op_time = period
                root.distance = 2
                root.is_leaf = 1
                list_root_second.append(root)
        return list_parent, list_root_second

    def __predict_by_model(self, data, init):
        if(init):
            pass


    def __add_to_database(self, init):
        pass


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
