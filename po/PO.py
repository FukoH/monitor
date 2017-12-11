#!/usr/bin/python
# -*- coding: utf-8 -*

# coding: utf-8
from sqlalchemy import Column, Float, Integer, String, text
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
metadata = Base.metadata


class IndexDef(Base):
    __tablename__ = 'index_def'

    pk_id = Column(Integer, primary_key=True)
    index_id = Column(String(9), nullable=False)
    index_name = Column(String(32), nullable=False)
    category = Column(String(8), nullable=False)
    is_effective = Column(Integer, nullable=False,  server_default=text("'1'"))
    is_significant = Column(Integer, nullable=False)


class MainData(Base):
    __tablename__ = 'main_data'

    pk_id = Column(Integer, primary_key=True)
    index_id = Column(String(9), nullable=False)
    op_time = Column(String(6), nullable=False)
    true_value = Column(Float(32))
    predict_value = Column(Float(32))
    last_month_value = Column(Float(32))
    last_year_value = Column(Float(32))
    percentage_difference = Column(Float(8))
    description = Column(String(10))
    upper_bound = Column(Float(32))
    lower_bound = Column(Float(32))
    is_abnormal = Column(Integer)


class Relation(Base):
    __tablename__ = 'relation'

    pk_id = Column(Integer, primary_key=True)
    op_time = Column(String(6), nullable=False)
    index_id = Column(String(9), nullable=False)
    parent_id = Column(String(9), nullable=False)
    distance = Column(Integer, nullable=False)
    is_leaf = Column(Integer, nullable=False)
    influence_factor = Column(Float(3))