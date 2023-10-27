import os
import argparse
import torch
import mlflow
from torch.utils.data import DataLoader

from app.application.source.train import (
    train,
    eval,
    visualize_predictions,
    read_dataframe,
    post_process_data,
    feature_engineer,
    split_data,
)
from app.application.source.dataloader import MyDataset
from app.config.constant import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")

    # Example of adding optional argument with a default value
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_visualize", action="store_true")

    # Parse the arguments
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    mlflow_endpoint = os.getenv('MLFLOW_ENDPOINT')
    # Configure the MLflow client to connect to your MLflow service.
    mlflow.set_tracking_uri(mlflow_endpoint)

    # Load data
    data = read_dataframe(PATH_DATA, checkpoint = INIT_HOURUTC_DATA_INGEST)
    data = post_process_data(data)
    X, y = feature_engineer(data)
    X_train, y_train, X_test, y_test = split_data(X, y)
    train_dataset = MyDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    test_dataset = MyDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    # Initialize MLflow
    mlflow.start_run()

    # Start to do actions
    if args.do_train:
        train(train_loader, NUM_EPOCH, DIR_SAVE_CKP, test_loader)
    if args.do_test:
        model = torch.load(os.path.join(DIR_SAVE_CKP, "best_model.pt"))
        score = eval(model, test_loader)
        mlflow.log_param("Test score", score)
    if args.do_visualize:
        model = torch.load(os.path.join(DIR_SAVE_CKP, "best_model.pt"))
        visualize_predictions(
            model, test_loader, name="test", path_save_plot=DIR_SAVE_IMG
        )
    # End MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()
