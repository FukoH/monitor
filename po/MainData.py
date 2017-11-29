#!/usr/bin/python
# -*- coding: utf-8 -*

class MainData:
    # __slots__ = ('pk_id',
    #              'index_pk_id',
    #              'op_time',
    #              'true_value',
    #              'predict_value',
    #              'last_month_value',
    #              'last_year_value',
    #              'percentage_difference',
    #              'description',
    #              'upper_bound',
    #              'lower_bound',
    #              'is_abnormal')
    @property
    def pk_id(self):
        return self.__pk_id

    @property
    def index_pk_id(self):
        return self.__index_pk_id

    @property
    def op_time(self):
        return self.__op_time

    @property
    def true_value(self):
        return self.__true_value

    @property
    def predict_value(self):
        return self.__predict_value

    @property
    def last_month_value(self):
        return self.__last_month_value

    @property
    def last_year_value(self):
        return self.__last_year_value

    @property
    def percentage_difference(self):
        return self.__percentage_difference

    @property
    def description(self):
        return self.__description

    @property
    def upper_bound(self):
        return self.__upper_bound

    @property
    def lower_bound(self):
        return self.__lower_bound

    @property
    def is_abnormal(self):
        return self.__is_abnormal

    @pk_id.setter
    def pk_id(self, pk_id):
        self.__pk_id = pk_id

    @index_pk_id.setter
    def index_pk_id(self, index_pk_id):
        self.__index_pk_id = index_pk_id

    @op_time.setter
    def op_time(self, op_time):
        self.__op_time = op_time

    @true_value.setter
    def true_value(self, true_value):
        self.__true_value = true_value

    @predict_value.setter
    def predict_value(self, predict_value):
        self.__predict_value = predict_value

    @last_month_value.setter
    def last_month_value(self, last_month_value):
        self.__last_month_value = last_month_value

    @last_year_value.setter
    def last_year_value(self, last_year_value):
        self.__last_year_value = last_year_value

    @percentage_difference.setter
    def percentage_difference(self, percentage_difference):
        self.__percentage_difference = percentage_difference

    @description.setter
    def description(self, description):
        self.__description = description

    @upper_bound.setter
    def upper_bound(self, upper_bound):
        self.__upper_bound = upper_bound

    @lower_bound.setter
    def lower_bound(self, lower_bound):
        self.__lower_bound = lower_bound

    @is_abnormal.setter
    def is_abnormal(self, is_abnormal):
        self.__is_abnormal = is_abnormal
