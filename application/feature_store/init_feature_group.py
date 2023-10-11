import pandas as pd 
import numpy as np

from controller.HopsworksController import FeatureStoreController
from application.source.train import post_process_data, transform


df = pd.read_csv('application/dataset/ConsumptionDE35Hour.txt',delimiter=";")
store = FeatureStoreController() 

df = post_process_data(df)
df = transform(df, hour_look_back=24)
df = df.loc[(df['last24']!=0)].reset_index(drop = True)
df = df[df['HourDK'] < '2023-06-01']
print (df)
df['eventtime'] = df['HourDK'].values.astype(np.int64)
store.create_feature_group(name='energy_consumption_fg', 
                           version=1,
                           primary_key=['eventtime', 'PriceArea', 'ConsumerType_DE35'],
                           event_time='eventtime',
                           online_enabled= False,
                           ingest_data= df
                           )
