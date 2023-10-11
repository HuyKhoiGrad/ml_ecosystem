import pandas as pd 
from datetime import datetime

from controller.HopsworksController import FeatureStoreController
from application.source.train import train
from application.utils.Logging import Logger

store = FeatureStoreController()

logger = Logger("Online training pipeline")

def create_online_training_data(feature_group_name: str, version: int, current_time: str):
    dt_obj = datetime.strptime(current_time,
                           '%d-%m-%Y %H:%M:%S')
    millisec = dt_obj.timestamp() * 1000
    feature_group = store.get_feature_group(feature_group_name, version)
    query = feature_group.filter(feature_group.event_time > millisec)
    feature_view = store.create_feature_view(
        name = 'feature_group_name',
        version=1,
        query=query,
        labels = ["totalcon"]
    )
    td_version, _ = store.create_train_test_dataset(feature_view=feature_view, name = current_time, test_size=0.2, data_format='csv')

    X_train, X_test, y_train, y_test = feature_view.get_train_test_split(td_version) 

    logger.info("Create online training data success")

    return X_train, X_test, y_train, y_test