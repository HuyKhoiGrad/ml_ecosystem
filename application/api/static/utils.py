import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torch.nn as nn


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
            # plt.show()
            plt.savefig(f"{path_save_plot}/{name}.png")
            break  # Visualize only the first batch of data


def read_content_file(file_contents: list) -> pd.DataFrame:
    contents = [i.strip().split(";") for i in file_contents]
    columns = contents[0]
    rows = contents[1:]
    df = pd.DataFrame(rows, columns=columns)
    return df
