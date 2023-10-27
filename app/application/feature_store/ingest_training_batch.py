import pandas as pd
import numpy as np
from datetime import datetime
from hsfs.feature import Feature

from app.controller.YugabyteController import YugaByteDBController
from app.config.DatabaseConfig import YugabyteConfig
from app.controller.HopsworksController import FeatureStoreController
from app.controller.RedisController import RedisOnlineStore
from app.config.RedisConfig import RedisClusterConnection
from app.config.constant import *

from app.application.source.train import transform 
from app.application.utils.Logging import Logger


logger = Logger("Ingest training batch data")

def get_source_data(curr_time: str, cols: list, source_db:YugaByteDBController) -> pd.DataFrame:
    sql = f"""SELECT * from energy.energy_consumption where hourutc + interval '1hour' = '{curr_time}' """
    source_data = source_db.get_data(sql = sql, col_name = cols)
    logger.info("Query source data success")
    return source_data

def ingest_batch_data(feature_group, data_source: pd.DataFrame):
    data = transform(data_source, 24)
    data['eventtime'] = data['HourUTC'].values.astype(np.int64)
    feature_group.insert(features=data)
    logger.info("Online training data ingest success")

def exec_ingest_source_data():
    # Init connection 
    store = FeatureStoreController()
    feature_group = store.get_feature_group(name = FEATURE_GROUP_ONL_NAME,
                                            version = FEATURE_GROUP_ONL_VERSION)
    source_db = YugaByteDBController(params= YugabyteConfig().get_config())
    source_db.connect() 

    redis = RedisOnlineStore(params = RedisClusterConnection().getConfig())
    redis.connect()
    
    # Get current data from source 
    latest_checkpoint = redis.getDataByKey(name = REDIS_INFERENCE_NAME, value = REDIS_INFERENCE_KEY)

    current_df = get_source_data(curr_time = latest_checkpoint, cols = SOURCE_DATA_COLUMNS, source_db = source_db)

    # Ingest current data to feature store
    ingest_batch_data(feature_group, data_source = current_df)