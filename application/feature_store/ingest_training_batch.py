import pandas as pd
import numpy as np
from datetime import datetime

from controller.YugabyteController import YugaByteDBController
from config.DatabaseConfig import YugabyteConfig
from controller.HopsworksController import FeatureStoreController
from application.source.train import transform 
from application.utils.Logging import Logger


logger = Logger("Ingest training batch data")

feature_group_name = "energy_consumption"
source_db = YugaByteDBController(params= YugabyteConfig().get_config())
source_db.connect() 
store = FeatureStoreController()

def get_source_data(curr_time: str, cols: list) -> pd.DataFrame:
    sql = f"""SELECT * from energy.data where "HourDK" + interval '1hour' = '{curr_time}' """
    source_data = source_db.get_data(sql = sql, col_name = cols)
    logger.info("Query source data success")
    return source_data

def ingest_batch_data(feature_group_name:str, fg_version:int, data_source: pd.DataFrame):
    data = transform(data_source, 24)
    data['eventtime'] = data['HourDK'].values.astype(np.int64)
    feature_group = store.get_feature_group(name = feature_group_name, version = fg_version) 
    feature_group.insert(features=data)
    logger.info("Online training data ingest success")




    



