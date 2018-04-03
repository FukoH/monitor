#!/data/quota/fanxiao/anaconda3/bin/python
# -*- coding: utf-8 -*

import argparse
from logger.config import config
import logging
from factor_analyser import DefaultAnalyser
from relation import all_index_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction intervals and influencing factor analysis.')
    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('period', help=r"the period up to which we train the model,should be like 'yyyyMM'.e.g. '201709'")
    group.add_argument("-a", action='store_true', help='train model for all indexes')
    group.add_argument("-index-id", metavar='index-N', nargs='+', help='train model on specified index(es) ')
    parser.add_argument('--init', action='store_true', help='discard old model, then train model basing on all data')
    args = parser.parse_args()
    # print args
    # 目标指标列表
    # all_index_ids = ['ZB1001001', 'ZB1001002', 'ZB1001003']
    
    if args.a:
        ids = all_index_ids
    else:
        ids = args.index_id
    
    logging.config.dictConfig(config)
    logger = logging.getLogger('simple')
    for index_id in ids:
        logger.info("start processing {}".format(index_id))
        analyser = DefaultAnalyser(index_id)
        analyser.analyse(period=args.period, init=args.init)

