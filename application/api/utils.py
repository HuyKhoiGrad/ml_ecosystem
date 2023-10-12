import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mlflow

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from application import config

matplotlib.use("Agg")


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_ = torch.tensor(self.X[idx], dtype=torch.float32)
        y_ = torch.tensor([self.y[idx]], dtype=torch.float32)
        return x_, y_


def eval(model, test_loader) -> float:
    criterion = RMSELoss()

    # Set the model to evaluation mode
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()

    # Calculate average loss for the test set
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


def visualize_predictions(model, data_loader, name, path_save_plot):
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            # Forward pass to get predictions
            outputs = model(x_batch)
            predictions = outputs.squeeze().numpy()  # Assuming predictions are 1D

            # Ground truth values (y_batch)
            ground_truth = y_batch.numpy()

            # Visualize the results using a line plot
            tmp = plt.figure(figsize=(10, 6))
            plt.plot(ground_truth, label="Ground Truth", marker="o", linestyle="-")
            plt.plot(predictions, label="Predictions", marker="o", linestyle="--")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.title(f"Model Predictions vs Ground Truth - hello Data")
            plt.legend()
            # plt.show()
            plt.savefig(f"{path_save_plot}/{name}.png")
            break  # Visualize only the first batch of data


def split_data(X, y, train_size=0.8):
    train_size = int(len(X) * train_size)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    return X_train, y_train, X_test, y_test


def post_process_data(df: pd.DataFrame) -> pd.DataFrame:
    df["HourUTC"] = pd.to_datetime(df["HourUTC"])
    df["HourDK"] = pd.to_datetime(df["HourDK"])
    df = df.sort_values(by="HourUTC")
    return df


def feature_engineer(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hour_look_back = 24
    for i in range(1, hour_look_back + 1):
        df[f"last{i}"] = df.groupby(["ConsumerType_DE35", "PriceArea"])[
            "TotalCon"
        ].shift(fill_value=0, periods=i)
    features_1 = [f"last{i}" for i in range(1, hour_look_back + 1)]
    # features_2 = ["ConsumerType_DE35", "PriceArea"]
    features_2 = []
    features = features_1 + features_2
    target = "TotalCon"
    X = df[features].values.astype(dtype=float)
    y = df[target].values.astype(dtype=float)
    return X, y


def read_content_file(file_contents: list) -> pd.DataFrame:
    # contents = [i.decode("utf-8").strip().split(";") for i in file_contents]
    contents = [i.strip().split(";") for i in file_contents]
    columns = contents[0]
    rows = contents[1:]
    df = pd.DataFrame(rows, columns=columns)
    return df


def model_mlflow_predict(X_loader, loaded_model):
    tmp: torch.Tensor = loaded_model.predict(X_loader)
    return tmp.squeeze()


def get_model_mlflow():
    mlflow.set_tracking_uri(config.MLFLOW_ENDPOINT)
    logged_model = f"runs:/{config.BEST_RUN_ID}/models"

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pytorch.load_model(logged_model)
    return loaded_model


def run_predict(file_contents: list):
    criterion = RMSELoss()
    df = read_content_file(file_contents)
    data = post_process_data(df)
    X, y = feature_engineer(data)
    test_dataset = MyDataset(X, y)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    total_loss = 0
    model_mlflow = get_model_mlflow()
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            outputs = model_mlflow_predict(x_batch, model_mlflow)

            # for calculate loss
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            # for visualize
            if batch_idx == len(test_loader) - 1:
                predictions = outputs.squeeze().numpy()  # Assuming predictions are 1D
                ground_truth = y_batch.numpy()
                plt.figure(figsize=(10, 6))
                plt.plot(ground_truth, label="Ground Truth", marker="o", linestyle="-")
                plt.plot(predictions, label="Predictions", marker="o", linestyle="--")
                plt.xlabel("Time Step")
                plt.ylabel("Value")
                plt.title(f"Model Predictions vs Ground Truth - Predict data")
                plt.legend()
                # plt.show()
                plt.savefig(f"{config.STATIC}/plot.png")

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss, "plot.png"


def get_init_X_1(data_specific: pd.DataFrame, hour_look_back: int = 24) -> list:
    df = data_specific.copy()
    for i in range(1, hour_look_back + 1):
        df[f"last{i}"] = df["TotalCon"].shift(fill_value=0, periods=i)
    features = [f"last{i}" for i in range(1, hour_look_back + 1)]
    target = "TotalCon"
    X = df[features].values.astype(dtype=float).tolist()
    # return torch.tensor(X[-1]).unsqueeze(0).to(torch.float32)
    return X[-1]


def get_init_X_2(data_specific: pd.DataFrame, hour_look_back: int = 24):
    df = data_specific.copy()
    for i in range(1, hour_look_back + 1):
        df[f"last{i}"] = df["TotalCon"].shift(fill_value=0, periods=i)
    features = [f"last{i}" for i in range(1, hour_look_back + 1)]
    target = "TotalCon"
    X = df[features].values.astype(dtype=float)
    y = df[target].values.astype(dtype=float)
    return X, y


def read_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    return df


def get_result_next_n_hours(
    n: int, device_type: int, area: str, data: pd.DataFrame
) -> list[float]:
    test_data = data[
        (data["ConsumerType_DE35"] == device_type) & (data["PriceArea"] == area)
    ].iloc[:25]
    init_X = get_init_X_1(test_data)
    model_mlflow = get_model_mlflow()
    result = []
    for i in range(n):
        x = init_X[-24:]
        x_batch = torch.tensor(x).unsqueeze(0).to(torch.float32)
        y_pred = model_mlflow_predict(x_batch, model_mlflow)
        init_X.append(y_pred.item())
        result.append(y_pred.item())
    return result


def get_result_pass_n_hours(
    n: int, device_type: int, area: str, data: pd.DataFrame
) -> tuple[list, list]:
    test_data = data[
        (data["ConsumerType_DE35"] == device_type) & (data["PriceArea"] == area)
    ].iloc[: n + 25]
    X, y = get_init_X_2(data_specific=test_data)
    x_batch = torch.tensor(X).to(torch.float32)
    model_mlflow = get_model_mlflow()
    y_pred = model_mlflow_predict(x_batch, model_mlflow)
    return y.tolist()[25:], y_pred.tolist()[25:]


def visualize_for_streamlit(future: list, pass_label: list, pass_pred: list):
    plt.figure(figsize=(10, 6))
    plt.plot(pass_label, label="Ground Truth", marker="o", linestyle="-")
    plt.plot(pass_pred + future, label="Predictions", marker="o", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Model Predictions vs Ground Truth - Predict data")
    plt.legend()
    # plt.show()
    plt.savefig(f"{config.STATIC}/plot.png")


def run_streamlit(
    area_choice: str,
    device_type_choice: int,
    number_last_hour: int,
    number_next_hour: int,
):
    data = read_dataframe(config.PATH_DATA)
    data = post_process_data(data)
    future = get_result_next_n_hours(
        n=number_next_hour,
        area=area_choice,
        device_type=device_type_choice,
        data=data,
    )
    pass_gt, pass_pred = get_result_pass_n_hours(
        n=number_last_hour,
        area=area_choice,
        device_type=device_type_choice,
        data=data,
    )
    visualize_for_streamlit(future=future, pass_label=pass_gt, pass_pred=pass_pred)


def inference_batch(df_input: pd.DataFrame) -> pd.DataFrame:
    model_mlflow = get_model_mlflow()
    x_df = df_input[[f"last{i}" for i in range(1, 24)] + ["TotalCon"]]
    x = x_df.values
    x_batch = torch.tensor(x).unsqueeze(0).to(torch.float32)
    y_pred = model_mlflow_predict(x_batch, model_mlflow)
    df_input["pred"] = y_pred.detach().numpy().tolist()
    return df_input


if __name__ == "__main__":
    df = read_dataframe("ConsumptionDE35Hour.txt")
    df = post_process_data(df)
    hour_look_back = 24
    for i in range(1, hour_look_back + 1):
        df[f"last{i}"] = df.groupby(["ConsumerType_DE35", "PriceArea"])[
            "TotalCon"
        ].shift(fill_value=0, periods=i)
    test_df = df.tail()
    result_df = inference_batch(df_input=test_df)
    print(result_df)
