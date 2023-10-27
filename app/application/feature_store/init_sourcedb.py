import pandas as pd
from datetime import datetime

from app.config.constant import *
from app.application.api.utils import post_process_data

from app.controller.YugabyteController import YugaByteDBController
from app.controller.RedisController import RedisOnlineStore
from app.config.DatabaseConfig import YugabyteConfig
from app.config.RedisConfig import RedisClusterConnection


df = pd.read_csv("app/application/dataset/ConsumptionDE35Hour.txt", delimiter=";")
df = post_process_data(df)

df = df[df["HourUTC"] <= INIT_HOURUTC_DATA_INGEST]

source_data = (
    df.groupby(["HourUTC", "HourDK", "PriceArea", "ConsumerType_DE35"])["TotalCon"]
    .sum()
    .reset_index()
)
source_data.to_csv("app/application/dataset/yb_source_data.csv", index=False)
