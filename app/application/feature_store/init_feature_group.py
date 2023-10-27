import pandas as pd
import numpy as np
from datetime import datetime

from app.controller.HopsworksController import FeatureStoreController
from app.controller.RedisController import RedisOnlineStore

# from app.config.RedisConfig import RedisClusterConnection

from app.application.feature_store.data_transform import create_batch_data
from app.config.constant import *


df = pd.read_csv("application/dataset/ConsumptionDE35Hour.txt", delimiter=";")
store = FeatureStoreController()

df = create_batch_data(df)
df = df[df["HourUTC"] <= INIT_HOURUTC_DATA_INGEST]
print(df)
df["eventtime"] = df["HourUTC"].values.astype(np.int64) // 1e6
df["eventtime"] = df["eventtime"].astype(int)
store.create_feature_group(
    name=FEATURE_GROUP_ONL_NAME,
    version=FEATURE_GROUP_ONL_VERSION,
    primary_key=["eventtime", "PriceArea", "ConsumerType_DE35"],
    event_time="eventtime",
    online_enabled=True,
    ingest_data=df,
)

redis = RedisOnlineStore()
redis.connect()
dt_obj = datetime.strptime(INIT_HOURUTC_DATA_INGEST, "%Y-%m-%d %H:%M:%S")
millisec = int(dt_obj.timestamp() * 1000)

redis.insertValueRedis(
    name=REDIS_INFERENCE_NAME, key=REDIS_INFERENCE_KEY, value=str(millisec)
)
