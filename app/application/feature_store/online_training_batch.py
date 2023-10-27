import pandas as pd
import time
from datetime import datetime

# from hsfs.feature import Feature

from app.config.constant import *
from app.application.source.train import train_1_batch
from app.application.utils.Logging import Logger

from app.controller.HopsworksController import FeatureStoreController
from app.controller.RedisController import RedisOnlineStore


logger = Logger("Online training pipeline")


def create_online_training_data(store: FeatureStoreController, current_time: str):
    fg_name = FEATURE_GROUP_ONL_NAME
    fg_version = FEATURE_GROUP_ONL_VERSION

    feature_group = store.get_feature_group(name=fg_name, version=fg_version)
    start_time = time.time()
    batch_data = (
        feature_group.select_all()
        .filter(
            (feature_group["eventtime"] <= current_time)
            & (feature_group["eventtime"] > current_time - 3600000 * 24)
        )
        .read()
    )

    cols = [f"last{i}" for i in range(1, 24)]
    cols.extend(["totalcon"])
    batch_data = batch_data[cols]

    return batch_data


def exec_online_training(**kwargs):
    store = FeatureStoreController()
    redis = RedisOnlineStore()
    redis.connect()

    exec_date = redis.getDataByKey(name=REDIS_INFERENCE_NAME, key=REDIS_INFERENCE_KEY)

    data = create_online_training_data(store, current_time=exec_date)

    # Run model training

    # Save model id to redis
