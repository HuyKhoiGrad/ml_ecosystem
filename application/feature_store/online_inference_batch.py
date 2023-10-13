import pandas as pd 
from datetime import datetime 
from hsfs.feature import Feature
import time

from config.constant import *
from application.utils.Logging import Logger

from controller.HopsworksController import FeatureStoreController
from controller.RedisController import RedisOnlineStore
from controller.YugabyteController import YugaByteDBController
from config.RedisConfig import RedisClusterConnection

logger = Logger("Get_online_inference_feature")

def init_feature_view(curr_time:str, feature_group, store, fv_name, fv_version):
    query = feature_group.filter((Feature("hourdk") == curr_time))
    feature_view = store.create_feature_view(
        name = fv_name,
        version= fv_version,
        query=query,
        label = ["totalcon"]
    )

    return feature_view

def get_batch_inference(store, curr_time, feature_group = None) -> list:
    fg_name = FEATURE_GROUP_ONL_NAME
    fg_version = FEATURE_GROUP_ONL_VERSION
    
    if feature_group == None:
        feature_group = store.get_feature_group(name = fg_name, version = fg_version)
    start_time = time.time()
    batch_data = feature_group.select_all().filter((Feature("eventtime") == curr_time)).read()

    cols = [f"last{i}" for i in range(1,24)]
    cols.extend(["totalcon"])
    batch_data = batch_data[cols]

    run_time = time.time() - start_time
    logger.info(f"Inference vector retrieve success in {run_time}")
    return batch_data

def ingest_predict_to_source(predict_df: pd.DataFrame, source_db: YugaByteDBController):
    source_db.insert_data(table_name='predict_consumption',
                          update_data= predict_df,
                          constraint_key='key')


def execute_batch_inference(**kwargs):
    store = FeatureStoreController()
    feature_group = store.get_feature_group(name = FEATURE_GROUP_ONL_NAME, version = FEATURE_GROUP_ONL_VERSION)

    redis_config = RedisClusterConnection().getConfig()
    redis = RedisOnlineStore(redis_config)
    redis.connect()
    exec_date = redis.getDataByKey(name= REDIS_INFERENCE_NAME, key = REDIS_INFERENCE_KEY)

    feature_vector = get_batch_inference(store, exec_date, feature_group)
    
    # Run model prediction 

    exec_date += 3600000
    redis.insertValueRedis(name= REDIS_INFERENCE_NAME, key = REDIS_INFERENCE_KEY, data = exec_date)



