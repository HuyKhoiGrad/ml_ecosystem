import pandas as pd

from app.controller.YugabyteController import YugaByteDBController
from app.config.DatabaseConfig import YugabyteConfig
from app.controller.RedisController import RedisOnlineStore
from app.config.RedisConfig import RedisClusterConnection
from app.config.constant import *

from app.application.utils.Logging import Logger


def exec_ingest_data_to_dwh(exec_datetime = None):
    msg = '' 
    try:
        redis_config = RedisClusterConnection().getConfig()
        redis = RedisOnlineStore(redis_config)
        redis.connect()

        datawh = YugaByteDBController(YugabyteConfig().get_config())
        datawh.connect()

        source_data = pd.read_csv('application/dataset/ConsumptionDE35Hour.txt',delimiter=";")
        # Get current datetime in redis
        if exec_datetime == None:
            exec_datetime = redis.getDataByKey(name= REDIS_INFERENCE_NAME, key = REDIS_INFERENCE_KEY)
        
        curr_data = source_data[source_data['HourUTC'] == exec_datetime]
        datawh.insert_data(table_name= SOURCE_TABLE, update_data= curr_data, constraint_key= SOURCE_CONSTRAINT)
        msg = 'Successful ingest data in {exec_datetime} to YugabyteDB'
    except Exception as e:
        msg = f'Error: {str(e)}'
    return msg
    