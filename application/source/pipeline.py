import os
import pandas as pd  
from datetime import datetime
from hsfs.feature import Feature
import torch
import mlflow
from torch.utils.data import DataLoader

from controller.HopsworksController import FeatureStoreController
from controller.RedisController import RedisOnlineStore
from config.RedisConfig import RedisClusterConnection
from config.constant import *
from application.source.train import train, split_data
from application.source.dataloader import MyDataset


mlflow_endpoint = os.getenv('MLFLOW_ENDPOINT')

def dataloader(df, hour_look_back):
    features_1 = [f"last{i}" for i in range(1, hour_look_back + 1)]
    features_2 = []
    features = features_1 + features_2
    target = "TotalCon"
    X = df[features].values.astype(dtype=float)
    y = df[target].values.astype(dtype=float)

    X_train, y_train, X_test, y_test = split_data(X, y)
    train_dataset = MyDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    test_dataset = MyDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return train_loader, test_loader

def train_pipeline():
    fg_name = FEATURE_GROUP_ONL_NAME
    fg_version = FEATURE_GROUP_ONL_VERSION

    mlflow.set_tracking_uri(mlflow_endpoint)
    mlflow.start_run()

    store = FeatureStoreController()
    
    feature_group = store.get_feature_group(name = fg_name, version = fg_version)

    dt_obj = datetime.strptime(INIT_HOURUTC_DATA_INGEST,
                           '%Y-%m-%d %H:%M:%S')
    current_time = int(dt_obj.timestamp() * 1000)
    batch_data = feature_group.select_all().filter((Feature("eventtime") <= current_time)).read()   
    train_loader, test_loader = dataloader(batch_data, hour_look_back = 24)
    train(train_loader, NUM_EPOCH, DIR_SAVE_CKP, test_loader)

    # End MLflow run
    mlflow.end_run()

if __name__ == '__main__':
    train_pipeline()