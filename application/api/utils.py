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
    contents = [i.decode("utf-8").strip().split(";") for i in file_contents]
    columns = contents[0]
    rows = contents[1:]
    df = pd.DataFrame(rows, columns=columns)
    return df


def model_mlflow_predict(X_loader):
    mlflow.set_tracking_uri(config.MLFLOW_ENDPOINT)
    logged_model = f"runs:/{config.BEST_RUN_ID}/models"

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pytorch.load_model(logged_model)
    tmp = loaded_model.predict(X_loader)
    return tmp


def run_predict(file_contents: list):
    criterion = RMSELoss()
    df = read_content_file(file_contents)
    data = post_process_data(df)
    X, y = feature_engineer(data)
    test_dataset = MyDataset(X, y)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            outputs = model_mlflow_predict(x_batch)

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
