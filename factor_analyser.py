#!/usr/bin/python
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn import linear_model, model_selection
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from po.PO import MainData
from po.PO import Relation
from DB.DBConnector import DBConnector


class FactorAnalyser(object):
    _target = []
    _factor_index_id = []
    _level_two_factor_id = {}  # {'index1':[index1-a,index1-b],'index2':[index2-a,index2-b]}
    connector = DBConnector()

    def analyse(self, period, init):
        """
        分析主方法
        :param period:  分析哪个期间
        :param init: 是否第一次初始化
        :return:
        """
        # load data
        first_level_data, second_level_data = self.__get_data_by_period(period)
        # predict
        self.__predict(second_level_data, period, init)
        # calculate factor
        dic = self.__modeling(first_level_data)
        _list = self.__save_result(dic, period, init)
        list_parent, list_root = self.__save_second_level(period, init)
        if list_parent:
            _list.extend(list_parent)
        if list_root:
            _list.extend(list_root)
        self.connector.add_data(_list)

    def __get_data_from_csv(self, path=r'./data/raw_data.csv'):
        """
        从原始的excel把数据转成关系型
        :param path:
        :return:
        """
        raw_data = pd.read_csv(path, encoding='gbk')
        raw_data['OP_TIME'].astype('int')
        # model_data = raw_data[['OP_TIME', 'INDEX_ID', 'INDEX_VALUE']][
        #     raw_data['INDEX_ID'].isin(self.__factor_index_id)]
        output = raw_data[['OP_TIME', 'INDEX_VALUE']][raw_data['INDEX_ID'].isin(self._target)]
        output.columns = ['OP_TIME', self._target[0]]
        for column in self._factor_index_id:
            short = raw_data[['OP_TIME', 'INDEX_VALUE']][raw_data['INDEX_ID'].isin([column])]
            short.columns = ['OP_TIME', column]
            output = output.merge(short, how='outer', on='OP_TIME')
        first_level_data = output
        index_set = set()
        for v in self._level_two_factor_id.values():
            for index in v:
                index_set.add(index)
        # second_level = raw_data[['OP_TIME', 'INDEX_VALUE']][raw_data['INDEX_ID'].isin(list(index_set).append('OP_TIME'))]
        # index_list = list(index_set)
        for column in index_set:
            short = raw_data[['OP_TIME', 'INDEX_VALUE']][raw_data['INDEX_ID'].isin([column])]
            short.columns = ['OP_TIME', column]
            output = output.merge(short, how='outer', on='OP_TIME')
        second_level_data = output
        return first_level_data, second_level_data

    def __get_data_by_period(self, period):
        """

        :param period:  生成哪个期间的数据
        :return:
        """
        period = int(period)
        first_level, second_level = self.__get_data_from_csv()
        fisrst_level_series = first_level[(first_level['OP_TIME'] <= period) & (first_level['OP_TIME'] >= 201303)]
        second_level_series = second_level[(second_level['OP_TIME'] <= period) & (second_level['OP_TIME'] >= 201303)]
        fisrst_level_series = fisrst_level_series.dropna()
        second_level_series = second_level_series.dropna()
        #second_level_series = second_level_series.sort_values('OP_TIME').reset_index().interpolate().dropna()
        #fisrst_level_series=fisrst_level_series.interpolate()
        #second_level_series=second_level_series.interpolate()
        return fisrst_level_series, second_level_series

    def __predict(self, data, period, init):
        data['OP_TIME'].astype('str')

        columns = data.columns
        columns = np.delete(columns, 0)
        for column in columns:
            maindata_list = []
            df = data[['OP_TIME', column]]
            df = df.sort_values('OP_TIME')
            df = df.reset_index()
            df = df.drop(['index'], axis=1)
            result = self.__predict_by_model(df, init, period)
            if not init:

                maindata = MainData()
                # maindata.index_pk_id = self.connector.session.query(IndexDef).filter_by(
                # index_name="column").one().index_id
                maindata.index_id = column
                maindata.op_time = period
                df = df.reset_index()
                # TODO 得到上一期的op_time
                maindata.last_period = df['OP_TIME'].iloc[df.index[df['OP_TIME'] == period] - 1].item()
                # maindata.true_value = df[column].iloc[df.index[df['OP_TIME']==period] - 1 ].item()
                maindata.true_value = df[column].iloc[df.index[df['OP_TIME'] == period]].item()
                last_period_maindata = self.connector.select_maindata(maindata)
                me = self.__get_me(df, last_period_maindata.predict_value)
                maindata.predict_value = result.item()
                maindata.upper_bound = last_period_maindata.predict_value + me
                maindata.lower_bound = last_period_maindata.predict_value - me
                maindata.last_month_value = last_period_maindata.predict_value

                maindata.last_month_true_value = last_period_maindata.true_value
                if last_period_maindata.true_value != None and last_period_maindata.true_value != 0:
                    maindata.last_month_percentage_difference = maindata.true_value / last_period_maindata.true_value - 1
                if not df['OP_TIME'].iloc[df.index[df['OP_TIME'] == str((int(period) - 100))]].empty:
                    maindata.last_period = df['OP_TIME'].iloc[df.index[df['OP_TIME'] == str((int(period) - 100))]].item()
                    last_year_maindata = self.connector.select_maindata(maindata)
                    maindata.last_year_value = last_year_maindata.true_value
                    maindata.percentage_difference = maindata.true_value/maindata.last_year_value - 1
                
                if maindata.true_value > maindata.upper_bound:
                    maindata.description = '>上限值'
                    maindata.is_abnormal = 1
                elif maindata.true_value < maindata.lower_bound:
                    maindata.description = '<下限值'
                    maindata.is_abnormal = 1
                else:
                    maindata.description = '预测区间内'
                    maindata.is_abnormal = 0
                maindata_list.append(maindata)
            else:

                result = result.flatten()
                i = 0  # counter
                last_month_true_value = 0  # 保存上月真实值
                for index, row in df.iterrows():
                    # print(row['name'], row['score'])
                    period = df.iloc[len(df) - 1]['OP_TIME']
                    # result = self.__predict_by_model(df, False, period)
                    if i == 0:
                        me = self.__get_me(df, result[i])
                    else:
                        me = self.__get_me(df, result[i - 1])
                    maindata = MainData()
                    maindata.index_id = column
                    maindata.op_time = str(row['OP_TIME'])[:-2]
                    maindata.true_value = float(row[column])
                    if i == 0:
                        maindata.last_month_value = maindata.true_value
                        maindata.last_month_true_value = last_month_true_value
                        last_month_true_value = maindata.true_value
                    else:
                        maindata.last_month_value = float(result[i - 1])
                        # 把上月真实值赋值并更新
                        maindata.last_month_true_value = last_month_true_value
                        last_month_true_value = maindata.true_value
                        maindata.last_month_percentage_difference = (
                                                                        maindata.true_value / maindata.last_month_true_value) - 1
                        if not df[column].iloc[df.index[df['OP_TIME'] == (int(row['OP_TIME'] - 100))]].empty:
                            # print(row['OP_TIME']-100)
                            maindata.last_year_value = df[column].iloc[
                                df.index[df['OP_TIME'] == (int(row['OP_TIME'] - 100))]].item()
                            maindata.percentage_difference = maindata.true_value / maindata.last_year_value - 1
                        maindata.upper_bound = float(result[i - 1] + me)
                        maindata.lower_bound = float(result[i - 1] - me)
                    maindata.predict_value = float(result[i])

                    if maindata.true_value > maindata.upper_bound:
                        maindata.description = '>上限值'
                        maindata.is_abnormal = 1
                    elif maindata.true_value < maindata.lower_bound:
                        maindata.description = '<下限值'
                        maindata.is_abnormal = 1
                    else:
                        maindata.description = '预测区间内'
                        maindata.is_abnormal = 0

                    maindata_list.append(maindata)
                    i = i + 1

            # return maindata_list

            self.__add_to_database(maindata_list, init)
            print('Added {} to database'.format(column))

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
        poly_X = data.drop(['OP_TIME', self._target[0]], axis=1)
        y = data[self._target[0]]
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
        _list = []
        for key, value in dic.iteritems():
            r = Relation()
            r.index_id = key
            r.influence_factor = value
            r.parent_id = self._target[0]
            r.op_time = period
            r.distance = 1
            r.is_leaf = 0
            _list.append(r)
        return _list

    def __save_second_level(self, period, init):
        list_parent = []  # 和父节点
        list_root_second = []  # 和根节点
        for key, value in self._level_two_factor_id.iteritems():
            for v in value:
                r = Relation()
                r.index_id = v
                r.influence_factor = None
                r.parent_id = key
                r.op_time = period
                r.distance = 1
                r.is_leaf = 1
                list_parent.append(r)

                root = Relation()
                root.index_id = v
                root.influence_factor = None
                root.parent_id = self._target[0]
                root.op_time = period
                root.distance = 2
                root.is_leaf = 1
                list_root_second.append(root)
        return list_parent, list_root_second

    def __predict_by_model(self, data, init, period):
        # if init:
        #     return data
        # else:
        return self.__predict_by_LSTM(data, period, init)

    def __predict_by_LSTM(self, data, period, init):
        sys.setrecursionlimit(1048576)
        """
        载入指定路径的数据，usecols=[1] 读取第二列
        """
        column_name = data.columns[1]
        # 取dataframe中的数值
        dataset = data[column_name].values
        # 将数值类型转换成浮点型
        dataset = dataset.astype('float32')

        """
        定义一个array的值转换成矩阵的函数
        """

        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        # 标准化数据
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = dataset.reshape(len(dataset), 1)
        dataset = scaler.fit_transform(dataset)

        train_size = int(len(dataset) - 1)
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        """
        将数据转换成模型需要的形状，X=t and Y=t+1
        """
        look_back = 1
        # from set_super_parameter import look_back
        trainX, trainY = create_dataset(train, look_back)
        # testX, testY = create_dataset(test, look_back)
        """
        将数据转换成模型需要的形状，[样本samples,时间步 time steps, 特征features]
        """
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        # testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        # textX = numpt.
        """
        搭建LSTM神经网络
        """
        sample = 4
        nb_epoch = 100
        optimizer = 'adam'
        model = Sequential()
        model.add(LSTM(sample, input_dim=look_back))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.fit(trainX, trainY, nb_epoch=nb_epoch, batch_size=1, verbose=2)
        #        model = load_model('my.h5')
        # t_0 = data[column_name][data['OP_TIME'] == period].item()  # 要预测的那一期

        if init:
            predict_value = scaler.inverse_transform(model.predict(dataset.reshape(len(dataset), 1, 1)))
            return predict_value
        else:
            data['OP_TIME'] = data['OP_TIME'].apply(str)
            t_0 = data.loc[data['OP_TIME'] == period, column_name]
            t_0 = scaler.transform(t_0.reshape(-1,1))
            predict_value = model.predict(t_0.reshape(1, 1, 1))
            return scaler.inverse_transform(predict_value)

    def __add_to_database(self, result, init):
        self.connector.add_data(result)
        # if init:
        #     pass
        # else:
        #     maindata = MainData()
        #     # maindata.index_pk_id =

    def __get_me(self, df, prediction):
        return prediction * 0.1


class BillUserAnalyser(FactorAnalyser):
    _target = ['ZB1001001']
    _factor_index_id = ['ZB1001101',
                        'ZB1001102',
                        'ZB1001103',
                        'ZB1001104',
                        'ZB1001105']
    _level_two_factor_id = {}


class NetBillUserAnalyser(FactorAnalyser):
    _target = ['ZB1001002']
    _factor_index_id = ['ZB1001106',
                        'ZB1001107',
                        'ZB1001108']
    _level_two_factor_id = {
            'ZB1001106':[],
            'ZB1001107':['ZB1001101',
                         'ZB1001102',
                        'ZB1001103',
                        'ZB1001104',
                        'ZB1001105',
                        'ZB1001206',
                        'ZB1001207'],
            'ZB1001108':['ZB1001101',
                         'ZB1001102',
                        'ZB1001103',
                        'ZB1001104',
                        'ZB1001105',
                        'ZB1001206',
                        'ZB1001207']
            } # {'index1':[index1-a,index1-b],'index2':[index2-a,index2-b]}


class FeeAnalyser(FactorAnalyser):
    _target = ['ZB1001003']
    _factor_index_id = ['ZB1001301',
                        'ZB1001302',
                        'ZB1001303',
                        'ZB1001304',
                        'ZB1001305',
                        'ZB1001306']
    _level_two_factor_id = {}


class FactorAnalyserFactory(object):
    @classmethod
    def get_analyser(cls, index_id):
        if index_id == 'ZB1001001':
            return BillUserAnalyser()
        elif index_id == 'ZB1001002':
            return NetBillUserAnalyser()
        elif index_id == 'ZB1001003':
            return FeeAnalyser()
