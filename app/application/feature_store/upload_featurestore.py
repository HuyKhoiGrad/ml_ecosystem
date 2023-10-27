import pandas as pd
import numpy as np
from datetime import datetime
#from hsfs.feature import Feature

from app.controller.YugabyteController import YugaByteDBController
from app.config.DatabaseConfig import YugabyteConfig
from app.controller.HopsworksController import FeatureStoreController
from app.controller.RedisController import RedisOnlineStore
from app.config.RedisConfig import RedisClusterConnection
from app.config.constant import *

from app.application.utils.Logging import Logger


logger = Logger("UploadFeatureToHopsworks")

def get_source_data(curr_time: str, cols: list, source_db:YugaByteDBController) -> pd.DataFrame:
    sql = f"""SELECT * from {SOURCE_TABLE} where hourutc + interval '1hour' = '{curr_time}' """
    source_data = source_db.get_data(sql = sql, col_name = cols)
    logger.info("Query source data success")
    return source_data

def transform(df: pd.DataFrame, hour_look_back: int = 24) -> pd.DataFrame:
    for i in range(1, hour_look_back + 1):
        df[f"last{i}"] = df.groupby(["consumertype_de35", "pricearea"])[
            "totalcon"
        ].shift(fill_value=0, periods=i)
    return df

def ingest_batch_data(feature_group, data_source: pd.DataFrame):
    data = transform(data_source, 24)
    data['eventtime'] = data['hourutc'].values.astype(np.int64)
    feature_group.insert(features=data)
    logger.info("Online training data ingest success")

def exec_ingest_source_data(latest_checkpoint = None):
    # Init connection 
    msg = ''
    try:
        store = FeatureStoreController()
        feature_group = store.get_feature_group(name = FEATURE_GROUP_ONL_NAME,
                                                version = FEATURE_GROUP_ONL_VERSION)
        source_db = YugaByteDBController(params= YugabyteConfig().get_config())
        source_db.connect() 

        redis = RedisOnlineStore(params = RedisClusterConnection().getConfig())
        redis.connect()
        
        # Get current data from source 
        if latest_checkpoint == None:
            latest_checkpoint = redis.getDataByKey(name = REDIS_INFERENCE_NAME, value = REDIS_INFERENCE_KEY)
        current_df = get_source_data(curr_time = latest_checkpoint, 
                                     cols = SOURCE_DATA_COLUMNS, 
                                     source_db = source_db)

        # Ingest current data to feature store
        ingest_batch_data(feature_group, data_source = current_df)
        msg = 'Ingest feature to feature store success'
    except Exception as e:
        msg = f'Error: {str(e)}'
    return msg