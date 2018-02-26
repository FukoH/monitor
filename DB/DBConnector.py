#!/usr/bin/python
# -*- coding: utf-8 -*

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from po.PO import MainData
from DB.config import db_develop
from DB.config import config


class DBConnector:
    # engine = create_engine('mysql+mysqlconnector://%s:%s@%s:3306/%s' % (
    #     db_develop['user'], db_develop['password'], db_develop['host'], db_develop['base']))
    engine = create_engine('mysql+mysqlconnector://%s:%s@%s:3306/%s' % (
        config['user'], config['password'], config['host'], config['base']))
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)

    # 创建session对象:
    session = DBSession()

    def add_data(self, data):
        try:
            self.session.bulk_save_objects(data)
        # 提交即保存到数据库:
            self.session.commit()
        except:
            self.session.rollback()
        # 关闭session:
        finally:
            self.session.close()

    def insert_ignore(self, data):
        data_list = [dict((key, value) for key, value in d.__dict__.items()
                    if not callable(value) and not key.startswith('__') and not key.startswith('_')) for d in data]
        try:
            for d in data_list:
                self.session.execute(MainData.__table__.insert().prefix_with('IGNORE'), d)
        # self.session.bulk_save_objects(data)
        # 提交即保存到数据库:
            self.session.commit()
        except:
            self.session.rollback()
        # 关闭session:
        finally:
            self.session.close()

    def select_maindata(self, maindata):
        maindata = self.session.query(MainData).filter(MainData.op_time == maindata.last_period,
                                                       MainData.index_id == maindata.index_id).one()
        self.session.close()
        return maindata
