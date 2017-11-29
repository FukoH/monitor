import mysql.connector
import config


class DBConnector:
    def __init__(self):

        conn = mysql.connector.connect(user=config['db']['user'], password=config['db']['password'], database='test',
                                   host=config['db']['host'])
        cursor = conn.cursor()

