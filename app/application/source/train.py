import pandas as pd
import torch
import mlflow
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from app.application.source.dataloader import MyDataset
from app.application.source.model import MyModel
from app.application.source.loss import RMSELoss
from app.config import constant


def get_model_mlflow(run_id):
    mlflow.set_tracking_uri(constant.MLFLOW_ENDPOINT)
    logged_model = f"runs:/{run_id}/models"

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pytorch.load_model(logged_model)
    return loaded_model


def read_dataframe(path: str, checkpoint) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    df = df[df["HourUTC"] <= checkpoint]
    return df


def post_process_data(df: pd.DataFrame) -> pd.DataFrame:
    df["HourUTC"] = pd.to_datetime(df["HourUTC"])
    df["HourDK"] = pd.to_datetime(df["HourDK"])
    df = df.sort_values(by="HourUTC")
    return df


def transform(df: pd.DataFrame, hour_look_back: int = 24) -> pd.DataFrame:
    for i in range(1, hour_look_back + 1):
        df[f"last{i}"] = df.groupby(["ConsumerType_DE35", "PriceArea"])[
            "TotalCon"
        ].shift(fill_value=0, periods=i)
    return df


def feature_engineer(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hour_look_back = 24
    df = transform(df=df, hour_look_back=hour_look_back)
    features_1 = [f"last{i}" for i in range(1, hour_look_back + 1)]
    features_2 = []
    features = features_1 + features_2
    target = "TotalCon"
    X = df[features].values.astype(dtype=float)
    y = df[target].values.astype(dtype=float)
    return X, y


def split_data(X, y, train_size=0.8):
    train_size = int(len(X) * train_size)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    return X_train, y_train, X_test, y_test


def train(train_loader, num_epochs, path_save_ckp, test_loader=None):
    # Initialize the model
    model = MyModel()

    # Define loss function and optimizer
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    check_loss = []
    best_model = copy.deepcopy(model)
    mlflow.log_param("Num epoch", num_epochs)
    mlflow.log_param("Lr", 0.001)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print batch information if desired
            if (batch_idx + 1) % 1000 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        check_loss.append(avg_loss)

        # evaluate
        if test_loader is not None:
            with torch.no_grad():
                for batch_idx_eval, (x_batch_eval, y_batch_eval) in enumerate(
                    test_loader
                ):
                    # Forward pass
                    outputs = model(x_batch_eval)
                    loss = criterion(outputs, y_batch_eval)

                    total_loss += loss.item()
            avg_loss_val = total_loss / len(test_loader)
            mlflow.log_metric("loss_eval", avg_loss_val, step=epoch + 1)

        # Save best checkpoint
        if avg_loss < best_val_loss:
            torch.save(
                model,
                f"{path_save_ckp}/best_model.pt",
            )
            best_model = copy.deepcopy(model)
            best_val_loss = avg_loss
            mlflow.pytorch.log_model(model, "models")

        # Log metrics and parameters with MLflow
        mlflow.log_metric("loss_train", avg_loss, step=epoch + 1)

        # Print epoch information
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return best_model


def eval(model, test_loader) -> float:
    criterion = RMSELoss()

    # Set the model to evaluation mode
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)  # Assuming RMSELoss is defined

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
            plt.show()
            plt.savefig(f"{path_save_plot}/{name}.png")
            mlflow.log_artifact(f"{path_save_plot}/{name}.png", "plots")
            break  # Visualize only the first batch of data


def train_1_batch(
    df_input,
    id,
):
    mlflow.set_tracking_uri(constant.MLFLOW_ENDPOINT)
    mlflow.start_run()
    x_df = df_input[[f"last{i}" for i in range(1, 25)]]
    x = x_df.values
    y = df_input["totalcon"].values
    X = torch.tensor(x).unsqueeze(0).to(torch.float32)
    y = torch.tensor(y).to(torch.float32)

    model = get_model_mlflow(id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = RMSELoss()
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    run = mlflow.active_run()
    if run:
        run_id = run.info.run_id
        mlflow.pytorch.log_model(model, "models")
        return run_id
    return None


if __name__ == "__main__":
    print("hello")
    df = read_dataframe(
        "/Users/nguyenthinhquyen/source/ml_ecosystem/ConsumptionDE35Hour.txt",
        checkpoint="2023-05-31 21:00:00",
    )
    df = df[:100]
    df = post_process_data(df)
    hour_look_back = 24
    df = transform(df=df, hour_look_back=hour_look_back)
    df = df.rename(columns={"TotalCon": "totalcon"})
    a = train_1_batch(df_input=df, id="4c3d9d7ef686428da6554f7399fbd4d6")
    print(a)
