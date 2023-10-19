import pandas as pd 

from app.application.source.train import post_process_data, transform


def create_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    df = post_process_data(df)
    df = transform(df, hour_look_back=24)
    df = df.loc[(df['last24']!=0)].reset_index(drop = True)
    return df
