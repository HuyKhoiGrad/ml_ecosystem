import requests
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd

from app.config.constant import *
from app.controller.RedisController import RedisOnlineStore

# from app.config.RedisConfig import RedisClusterConnection


def run_inference():
    api = "http://localhost:5050/inference"

    redis = RedisOnlineStore()
    redis.connect()
    if redis.checkExists(REDIS_INFERENCE_NAME, REDIS_INFERENCE_KEY) == True:
        exec_date = redis.getDataByKey(REDIS_INFERENCE_NAME, REDIS_INFERENCE_KEY)

    data = {"current_time": exec_date}
    payload = json.dumps(data)
    resp = requests.post(
        api, data=payload, headers={"Content-Type": "application/json"}
    )
    print(resp)


def run_ingest():
    api = "http://localhost:5050/ingest"

    redis = RedisOnlineStore()
    redis.connect()
    if redis.checkExists(REDIS_INFERENCE_NAME, REDIS_INFERENCE_KEY) == True:
        exec_date = redis.getDataByKey(REDIS_INFERENCE_NAME, REDIS_INFERENCE_KEY)

    data = {"current_time": exec_date}
    payload = json.dumps(data)
    resp = requests.post(
        api, data=payload, headers={"Content-Type": "application/json"}
    )
    print(resp)


def run_upload():
    api = "http://localhost:5050/ingest"

    redis = RedisOnlineStore()
    redis.connect()
    if redis.checkExists(REDIS_INFERENCE_NAME, REDIS_INFERENCE_KEY) == True:
        exec_date = redis.getDataByKey(REDIS_INFERENCE_NAME, REDIS_INFERENCE_KEY)

    data = {"current_time": exec_date}
    payload = json.dumps(data)
    resp = requests.post(
        api, data=payload, headers={"Content-Type": "application/json"}
    )
    print(resp)


with DAG(
    dag_id="run_ingest",
    start_date=datetime.now(),
    schedule_interval="* 2 * * *",
    catchup=True,
) as dag:
    demo_task = PythonOperator(task_id="ingest", python_callable=run_ingest)
