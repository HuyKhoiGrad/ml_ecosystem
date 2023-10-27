import os

# Directory
root = os.getcwd()
PATH_DATA = f"{root}/app/application/dataset/ConsumptionDE35Hour.txt"
DIR_SAVE_CKP = f"{root}/app/application/weight"
DIR_SAVE_CKP_ONLINE = f"{root}/app/application/checkpoints_online"
DIR_SAVE_IMG = f"{root}/app/application/images"

# Hyper parameter
NUM_EPOCH = 10

# MLflow
MLFLOW_ENDPOINT = "http://localhost:5000"
BEST_RUN_ID = "4c3d9d7ef686428da6554f7399fbd4d6"

# API
STATIC = f"{root}/app/application/api/static"
