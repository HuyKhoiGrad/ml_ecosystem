import pandas as pd 
import numpy as np

from controller.HopsworksController import FeatureStoreController
from controller.RedisController import RedisOnlineStore
from config.RedisConfig import RedisClusterConnection

from .data_transform import create_batch_data
from config.constant import *


df = pd.read_csv('application/dataset/ConsumptionDE35Hour.txt',delimiter=";")
store = FeatureStoreController() 

df = create_batch_data(df)
df = df[df['HourDK'] <= INIT_HOURDK_DATA_INGEST]
print (df)
df['eventtime'] = df['HourDK'].values.astype(np.int64)
store.create_feature_group(name='energy_consumption_onl_fg', 
                           version=1,
                           primary_key=['eventtime', 'PriceArea', 'ConsumerType_DE35'],
                           event_time='eventtime',
                           online_enabled= True,
                           ingest_data= df
                           )

redis = RedisOnlineStore(params = RedisClusterConnection().getConfig())
redis.connect()
redis.insertValueRedis(name = REDIS_INFERENCE_NAME, version = REDIS_INFERENCE_KEY, value = str(INIT_HOURDK_DATA_INGEST))
