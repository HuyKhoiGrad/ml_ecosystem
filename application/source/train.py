import pandas as pd
import torch
import mlflow
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader import MyDataset
from model import MyModel
from loss import RMSELoss
from application import config


def read_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    return df


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
    x_batch,
    y_batch,
    model,
    optimizer,
    criterion,
    path_save_ckp,
    id: str = "",
    best_val_loss: float = 10**6,
):
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if loss.item() < best_val_loss:
        torch.save(
            model,
            f"{path_save_ckp}/best_model.pt",
        )
        best_model = copy.deepcopy(model)
        best_val_loss = loss.item()
        mlflow.pytorch.log_model(model, "models_online")
    torch.save(
        model,
        f"{path_save_ckp}/{id}.pt",
    )


if __name__ == "__main__":
    model = torch.load("application/checkpoints/best_model.pt")
    # model.eval()
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data = read_dataframe(config.PATH_DATA)
    data = post_process_data(data)
    X, y = feature_engineer(data)
    X_train, y_train, X_test, y_test = split_data(X, y)
    train_dataset = MyDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        train_1_batch(
            x_batch=x_batch,
            y_batch=y_batch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            path_save_ckp=config.DIR_SAVE_CKP_ONLINE,
            id="0",
        )
        break
