FEATURE_GROUP_NAME = 'energy_consumption_fg'
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_ONL_NAME = 'energy_consumption_onl_utceventtime'
FEATURE_GROUP_ONL_VERSION = 1
# FEATURE_VIEW_INF_NAME = 'energy_consumption_onl_view'
# FEATURE_VIEW_INF_VERSION = 1
# FEATURE_VIEW_ONL_NAME = ''
# FEATURE_VIEW_ONL_VERSION = ''

SOURCE_DATA_COLUMNS = ['hourutc', 'hourdk', 'pricearea', 'consumertype_de35', 'totalcon']

INIT_HOURUTC_DATA_INGEST = '2023-05-31 21:00:00'

REDIS_INFERENCE_NAME = 'checkpoint'
REDIS_INFERENCE_KEY = 'latest_date'

PATH_DATA = "app/application/dataset/ConsumptionDE35Hour.txt"
DIR_SAVE_CKP = "app/application/weight"
DIR_SAVE_CKP_ONLINE = "app/application/checkpoints_online"
DIR_SAVE_IMG = "app/application/images"

# Hyper parameter
NUM_EPOCH = 10

# MLflow
MLFLOW_ENDPOINT = "http://localhost:5000"
BEST_RUN_ID = "96643ba262e9442998a5e1a6cfc622f5"

# API
STATIC = "app/application/api/static"

# Yugabyte db
SOURCE_TABLE = 'energy.energy_consumption'
SOURCE_CONSTRAINT = 'energy_consumption_pk'
PREDICT_TABLE = 'energy.energy_predict'
PREDICT_CONSTRAINT = 'energy_predict_pk'
