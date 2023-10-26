import json
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.application.feature_store.online_inference_batch import execute_batch_inference
from app.application.feature_store.ingest_training_batch import exec_ingest_source_data


class runtime(BaseModel):
    current_time: str
class ResponseModel(BaseModel):
    msg: str = None

app = FastAPI()

@app.post("/inference", response_model=ResponseModel, status_code=200)
def inferer_batch(payload: runtime):
    curr_time = payload.current_time
    msg = execute_batch_inference(exec_datetime = curr_time)
    response = ResponseModel()
    response.msg = msg
    return json.dumps(response.__dict__)

@app.post("/ingest", response_model=ResponseModel, status_code=200)
def ingest_data_to_featurestore(payload: runtime):
    curr_time = payload.current_time
    msg = exec_ingest_source_data(latest_checkpoint = curr_time)
    response = ResponseModel()
    response.msg = msg
    return json.dumps(response.__dict__)