import pandas as pd 
from datetime import datetime 
from hsfs.feature import Feature
import time

from config.constant import *
from application.utils.Logging import Logger

from controller.HopsworksController import FeatureStoreController
from controller.RedisController import RedisOnlineStore
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
    start_time = time.time()
    if feature_group == None:
        feature_group = store.get_feature_group(name = fg_name, version = fg_version)

    batch_data = feature_group.select_all().filter((Feature("eventtime") == curr_time)).read()

    cols = [f"last{i}" for i in range(23,0,-1)]
    cols.extend(["totalcon"])
    batch_data = batch_data[cols]
    feature_vector = batch_data.values.tolist()

    run_time = time.time() - start_time
    logger.info(f"Inference vector retrieve success in {run_time}")

def execute_batch_inference(**kwargs):
    store = FeatureStoreController()
    feature_group = store.get_feature_group(name = FEATURE_GROUP_ONL_NAME, version = FEATURE_GROUP_ONL_VERSION)

    redis_config = RedisClusterConnection().getConfig()
    redis = RedisOnlineStore(redis_config)
    redis.connect()
    
    exec_date = redis.getDataByKey(name= REDIS_INFERENCE_NAME, key = REDIS_INFERENCE_KEY)

    get_batch_inference(store, exec_date, feature_group)



