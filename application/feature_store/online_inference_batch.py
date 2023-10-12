import pandas as pd 
from datetime import datetime 
from hsfs.feature import Feature
import time

from config.constant import *
from application.utils.Logging import Logger

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

def get_batch_inference(store, curr_time, feature_store = None) -> list:
    fg_name = FEATURE_GROUP_ONL_NAME
    fg_version = FEATURE_GROUP_ONL_VERSION
    start_time = time.time()
    if feature_store == None:
        feature_store = store.get_feature_group(name = fg_name, version = fg_version)

    batch_data = feature_store.select_all().filter((Feature("hourdk") == curr_time)).read()

    cols = [f"last{i}" for i in range(1,24,-1)]
    cols.extend(["totalcon"])
    batch_data = batch_data[cols]
    feature_vector = batch_data.values.tolist()

    run_time = time.time() - start_time
    logger.info(f"Inference vector retrieve success in {run_time}")

    return feature_vector

