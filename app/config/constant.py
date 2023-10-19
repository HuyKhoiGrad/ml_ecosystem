FEATURE_GROUP_NAME = 'energy_consumption_fg'
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_ONL_NAME = 'energy_consumption_onl_utceventtime'
FEATURE_GROUP_ONL_VERSION = 1
# FEATURE_VIEW_INF_NAME = 'energy_consumption_onl_view'
# FEATURE_VIEW_INF_VERSION = 1
# FEATURE_VIEW_ONL_NAME = ''
# FEATURE_VIEW_ONL_VERSION = ''

SOURCE_DATA_COLUMNS = ['"HourUTC"', '"HourDK"', '"PriceArea"', '"ConsumerType_DE35"', '"TotalCon"']

INIT_HOURUTC_DATA_INGEST = '2023-05-31 21:00:00'

REDIS_INFERENCE_NAME = 'checkpoint'
REDIS_INFERENCE_KEY = 'latest_date'

PATH_DATA = "application/dataset/ConsumptionDE35Hour.txt"
DIR_SAVE_CKP = "application/weight"
DIR_SAVE_CKP_ONLINE = "application/checkpoints_online"
DIR_SAVE_IMG = "application/images"

# Hyper parameter
NUM_EPOCH = 10

# MLflow
MLFLOW_ENDPOINT = "http://localhost:5000"
BEST_RUN_ID = "b1006048238748c893ab0ed7f43b6433"

# API
STATIC = "application/api/static"