#!/usr/bin/python
# -*- coding: utf-8 -*

from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# from po.PO import MainData
import config


class DBConnector:
    engine = create_engine('mysql+mysqlconnector://%s:%s@%s:3306/%s' % (
        config['user'], config['password'], config['host'], config['base']))
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)

    # 创建session对象:
    session = DBSession()

    def add_data(self, data):
        # 创建新User对象:
        # new_user = User(id='5', name='Bob')
        # 添加到session:
        # self.session.add(data)
        self.session.bulk_save_objects(data)
        # 提交即保存到数据库:
        self.session.commit()
        # 关闭session:
        # self.session.close()


        # def update_data(self,_class,data):
        #
        # # 创建新User对象:
        # # new_user = User(id='5', name='Bob')
        #     data = self.session.query(_class).filter_by(name="user1").first()
        #     data.password = "newpassword"
        #     session.commit()
        # conn = mysql.connector.connect(user=config['db']['user'], password=config['db']['password'], database='test',
        #                                host=config['db']['host'])
        # cursor = conn.cursor()

        # def __init__(self):

        # conn = mysql.connector.connect(user=config['db']['user'], password=config['db']['password'], database='test',
        #                            host=config['db']['host'])
        # cursor = conn.cursor()

# class MainDataDB(DBConnector):
#
#     def insert(self):
