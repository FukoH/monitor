#!/usr/bin/python
# -*- coding: utf-8 -*

import argparse

from factor_analyser import DefaultAnalyser

parser = argparse.ArgumentParser(description='Prediction intervals and influencing factor analysis.')
group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('period', help=r"the period up to which we train the model,should be like 'yyyyMM'.e.g. '201709'")
group.add_argument("-a", action='store_true', help='train model for all indexes')
group.add_argument("-index-id", metavar='index-N', nargs='+', help='train model on specified index(es) ')
parser.add_argument('--init', action='store_true', help='discard old model, then train model basing on all data')
args = parser.parse_args()
# print args
# 目标指标列表
all_index_ids = ['ZB1001001', 'ZB1001002', 'ZB1001003']

if args.a:
    ids = all_index_ids
else:
    ids = args.index_id

for index_id in ids:
    # analyser = FactorAnalyserFactory.get_analyser(index_id)
    analyser = DefaultAnalyser(index_id)
    analyser.analyse(period=args.period, init=args.init)

    # data_fetcher = DataFetcher()
    # raw_data = data_fetcher.get_data_up_to_period(args.period)
    # model = Model()
    # output_data = model.get_output_data()
    # data_saver = DataSaver()
    # data_saver.save(output_data)
# print args.index_id
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
