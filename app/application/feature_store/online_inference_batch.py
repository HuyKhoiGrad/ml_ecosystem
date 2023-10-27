import os 
import pandas as pd 
from datetime import datetime 
from hsfs.feature import Feature
import time

from app.config.constant import *
from app.application.api.utils import get_model_mlflow, inference_batch, datetime2milli, milli2datetime
from app.application.utils.Logging import Logger

from app.controller.HopsworksController import FeatureStoreController
from app.controller.RedisController import RedisOnlineStore
from app.config.RedisConfig import RedisClusterConnection
from app.controller.YugabyteController import YugaByteDBController
from app.config.DatabaseConfig import YugabyteConfig


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
    run_time = time.time() - start_time
    logger.info(f"Inference vector retrieve success in {run_time}")
    return batch_data

def ingest_predict_to_source(predict_df: pd.DataFrame, source_db: YugaByteDBController, key: str):
    source_db.insert_data(table_name='energy.predict_consumption',
                          update_data= predict_df,
                          constraint_key= key)


def execute_batch_inference(**kwargs):
    # Init connection 
    store = FeatureStoreController()
    feature_group = store.get_feature_group(name = FEATURE_GROUP_ONL_NAME, version = FEATURE_GROUP_ONL_VERSION)

    redis_config = RedisClusterConnection().getConfig()
    redis = RedisOnlineStore(redis_config)
    redis.connect()

    source_db = YugaByteDBController(YugabyteConfig().get_config())
    source_db.connect()

    # Get current datetime in redis
    exec_datetime = redis.getDataByKey(name= REDIS_INFERENCE_NAME, key = REDIS_INFERENCE_KEY)
    exec_millisec = datetime2milli(datestr = exec_datetime)

    # Get feature vector from feature store
    feature_vector = get_batch_inference(store, exec_millisec, feature_group)
    
    # Run model prediction  
    mlflow_endpoint = os.getenv('MLFLOW_ENDPOINT')
    model = get_model_mlflow(end_point=mlflow_endpoint, run_id = BEST_RUN_ID)
    predict_df = inference_batch(feature_vector, model)

    # Save predict to source db
    predict_df['hourdk'] = predict_df['hourdk'] + pd.Timedelta(hours=1)
    predict_df = predict_df[['hourdk','pricearea','consumertype_de35','pred']]
    predict_df['unique_key'] = predict_df['hourdk'].dt.strftime('%Y-%m-%d %H:%M:%S') + '-' + predict_df['pricearea'] + '-' + predict_df['consumertype_de35'].astype(str)
    ingest_predict_to_source(predict_df, source_db, key = 'unique_key')

    # Insert newkey to redis
    next_exec_millisec = exec_millisec + 3600000
    next_exec_time = milli2datetime(next_exec_millisec)
    redis.insertValueRedis(name= REDIS_INFERENCE_NAME, key = REDIS_INFERENCE_KEY, data = str(next_exec_time))

if __name__ == '__main__':
    execute_batch_inference()

