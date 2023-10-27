from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd

from app.config.constant import *
from app.controller.RedisController import RedisOnlineStore

# from app.config.RedisConfig import RedisClusterConnection


def add_date_redis():
    print("DEMO RUN: REDIS TEST CONNECT")
    redis = RedisOnlineStore()
    redis.connect()
    if redis.checkExists(REDIS_INFERENCE_NAME, REDIS_INFERENCE_KEY) == True:
        exec_date = redis.getDataByKey(REDIS_INFERENCE_NAME, REDIS_INFERENCE_KEY)
        exec_date += 3600000
    else:
        dt_obj = datetime.strptime(INIT_HOURUTC_DATA_INGEST, "%Y-%m-%d %H:%M:%S")
        exec_date = int(dt_obj.timestamp() * 1000)
    redis.insertValueRedis(
        name=REDIS_INFERENCE_NAME, key=REDIS_INFERENCE_KEY, data=exec_date
    )


with DAG(
    dag_id="demo",
    start_date=datetime.now(),
    schedule_interval="* 2 * * *",
    catchup=True,
) as dag:
    demo_task = PythonOperator(task_id="demo", python_callable=add_date_redis)
