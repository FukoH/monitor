#!/usr/bin/python
# -*- coding: utf-8 -*

from sqlalchemy import  create_engine
from sqlalchemy.orm import sessionmaker


from po.PO import MainData
from DB.config import db_develop
from DB.config import db

class DBConnector:
    # engine = create_engine('mysql+mysqlconnector://%s:%s@%s:3306/%s' % (
    #     db_develop['user'], db_develop['password'], db_develop['host'], db_develop['base']))
    engine = create_engine('mysql+mysqlconnector://%s:%s@%s:3306/%s' % (
        db['user'], db['password'], db['host'], db['base']))
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)

    # 创建session对象:
    session = DBSession()

    def add_data(self, data):

        self.session.bulk_save_objects(data)
        # 提交即保存到数据库:
        self.session.commit()
        # 关闭session:
        self.session.close()

    def select_maindata(self,maindata):
        maindata = self.session.query(MainData).filter(MainData.op_time == maindata.last_period,MainData.index_id==maindata.index_id).one()
        self.session.close()
        return maindata

