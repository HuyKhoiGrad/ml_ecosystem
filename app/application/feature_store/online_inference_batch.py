import os
import pandas as pd
from datetime import datetime

# from hsfs.feature import Feature
import time

from app.config.constant import *
from app.application.api.utils import (
    get_model_mlflow,
    inference_batch,
    datetime2milli,
    milli2datetime,
)
from app.application.utils.Logging import Logger

from app.controller.HopsworksController import FeatureStoreController
from app.controller.RedisController import RedisOnlineStore

# from app.config.RedisConfig import RedisClusterConnection
from app.controller.YugabyteController import YugaByteDBController

# from app.config.DatabaseConfig import YugabyteConfig


logger = Logger("Get_online_inference_feature")


def init_feature_view(curr_time: str, feature_group, store, fv_name, fv_version):
    query = feature_group.filter((feature_group["hourutc"] == curr_time))
    feature_view = store.create_feature_view(
        name=fv_name, version=fv_version, query=query, label=["totalcon"]
    )

    return feature_view


def get_batch_inference(store, curr_time, feature_group=None) -> list:
    fg_name = FEATURE_GROUP_ONL_NAME
    fg_version = FEATURE_GROUP_ONL_VERSION

    if feature_group == None:
        feature_group = store.get_feature_group(name=fg_name, version=fg_version)
    start_time = time.time()
    batch_data = (
        feature_group.select_all()
        .filter((feature_group["eventtime"] == curr_time))
        .read()
    )
    run_time = time.time() - start_time
    logger.info(f"Inference vector retrieve success in {run_time}")
    return batch_data


def ingest_predict_to_source(
    predict_df: pd.DataFrame, source_db: YugaByteDBController, key: str
):
    source_db.insert_data(
        table_name=PREDICT_TABLE,
        update_data=predict_df,
        constraint_key=PREDICT_CONSTRAINT,
    )


def execute_batch_inference(exec_datetime=None):
    msg = ""
    try:
        # Init connection
        store = FeatureStoreController()
        feature_group = store.get_feature_group(
            name=FEATURE_GROUP_ONL_NAME, version=FEATURE_GROUP_ONL_VERSION
        )

        redis = RedisOnlineStore()
        redis.connect()

        source_db = YugaByteDBController()
        source_db.connect()

        # Get current datetime in redis
        if exec_datetime == None:
            exec_datetime = redis.getDataByKey(
                name=REDIS_INFERENCE_NAME, key=REDIS_INFERENCE_KEY
            )
        exec_millisec = datetime2milli(datestr=exec_datetime)

        # Get feature vector from feature store
        feature_vector = get_batch_inference(store, exec_millisec, feature_group)

        # Run model prediction
        mlflow_endpoint = os.getenv("MLFLOW_ENDPOINT")
        model = get_model_mlflow(end_point=mlflow_endpoint, run_id=BEST_RUN_ID)
        predict_df = inference_batch(feature_vector, model)

        # Save predict to source db
        predict_df["hourutc"] = predict_df["hourutc"] + pd.Timedelta(hours=1)
        predict_df["hourdk"] = predict_df["hourdk"] + pd.Timedelta(hours=1)
        predict_df = predict_df[
            ["hourutc", "hourdk", "pricearea", "consumertype_de35", "predict"]
        ]
        ingest_predict_to_source(predict_df, source_db)

        # Insert newkey to redis
        next_exec_millisec = exec_millisec + 3600000
        next_exec_time = milli2datetime(next_exec_millisec)
        redis.insertValueRedis(
            name=REDIS_INFERENCE_NAME, key=REDIS_INFERENCE_KEY, data=str(next_exec_time)
        )
        msg = "Run batch inference susecessful"
    except Exception as e:
        msg = f"Error: {str(e)}"
    return msg


if __name__ == "__main__":
    execute_batch_inference()
