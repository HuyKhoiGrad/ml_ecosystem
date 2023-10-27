import hopsworks 
import os
import pandas as pd 

class FeatureStoreController():
    def __init__(self):
        api_key = os.getenv("HOPSWORKS_API_KEY", default="")
        self.project = hopsworks.login(api_key_value= api_key)
        self.fs = self.project.get_feature_store()

    def create_feature_group(self, name:str, version:int, primary_key: list, event_time: str, online_enabled: bool, ingest_data: pd.DataFrame):
        feature_group = self.fs.get_or_create_feature_group(
            name = name,
            version = version,
            primary_key = primary_key,
            event_time = event_time,
            online_enabled = online_enabled,
            stream= True
        )
        feature_group.insert(ingest_data) 
    
    def get_feature_group(self, name:str, version:int):
        feature_group = self.fs.get_feature_group(
            name = name,
            version = version
        )
        return feature_group
    
    def get_feature_view(self, name:str, version:int):
        feature_view = self.fs.get_feature_view(name = name, version = version)
        return feature_view
    
    def create_feature_view(self, name: str, version:int, label: list, query):
        feature_view = self.fs.get_or_create_feature_view(
            name = name,
            version=version,
            query=query,
            labels = label
        ) 
        return feature_view
    
    def create_eventtime_train_test_split(self, feature_view, name:str, coalesce:float, data_format:int, train_time:tuple, test_time: tuple): 
        train_start, train_end = train_time 
        test_start, test_end = test_time

        td_version, td_job = feature_view.create_train_test_split(
            description = f'{name} training dataset',
            train_start = train_start,
            train_end = train_end,
            test_start = test_start, 
            test_end = test_end,
            data_format = data_format,
            coalesce = coalesce
        )
        return td_version, td_job
        
    def create_train_test_dataset(self, feature_view, name, test_size:float, data_format:str):
        td_version, td_job = feature_view.create_train_test_split(
            description = f'{name} training dataset',
            data_format = data_format,
            test_size = test_size
        )
        return td_version, td_job
    
    def get_train_test_dataset(self, feature_view, td_version):
        X_train, X_test, y_train, y_test = feature_view.get_train_test_split(td_version)
        return X_train, X_test, y_train, y_test

    def del_feature_view(self, feature_view_name:str, version:int):  
        feature_view = self.fs.get_feature_view(name=feature_view_name, version=version)
        feature_view.delete()

if __name__ == '__main__':
    store = FeatureStoreController()
    store.del_feature_view(feature_view_name='energy_consume_view_batch', version = 1)