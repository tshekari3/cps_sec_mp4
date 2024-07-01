import torch
import math
import logging
import argparse
from data_reader import DataReader
from power_system_nn import PowerSystemNN
import os
from evaluator import Evaluator
from joblib import load

def main(data_folder, model_folder, dataset_type) -> None:
    """
    Main function to evaluate the neural network model on a test dataset.

    This function performs the following steps:
    1. Load and preprocess test data: Utilizes the DataReader class to load test features and labels. It also loads a pre-trained scaler and a list of selected features to preprocess the data correctly.
    2. Load the model: Initializes a PowerSystemNN model with dimensions derived from the preprocessed data, then loads the model's state from a saved state dictionary.
    3. Evaluate the model: Uses the Evaluator class to assess the model's performance on the test data.

    The paths to the data, model, scaler, and selected features are inputs.

    Returns:
        None
    """
    dataset_type = dataset_type.lower()
    feature_file_path = data_folder + f"/test_{dataset_type}_features.csv"
    label_file_path = data_folder + f"/test_{dataset_type}_labels.csv"
    scaler_file_path = model_folder + "/scaler.joblib"
    model_file_path = model_folder + "/model.pth"
    selected_features_file_path = model_folder + "/selected_feature_columns.txt"

    # 1. Load and preprocess data
    test_data_reader = DataReader(feature_file_path, label_file_path)
    # load the scaler
    scaler = load(scaler_file_path)
    # load the selected features
    with open(selected_features_file_path, 'r') as file:
        selected_feature_columns = file.read().splitlines()
    test_data_reader.load_test_data(scaler, selected_feature_columns)

    # 2. Load the model state
    model = PowerSystemNN(input_dim = test_data_reader.feature_dim, output_dim = test_data_reader.label_dim)
    model.load_state_dict(torch.load(model_file_path))

    # 3. Evaluate the model
    evaluator = Evaluator(model)
    avg_accuracy, avg_f1 = evaluator.evaluate(test_data_reader)

    X = accuracy_to_x(avg_accuracy)
    Y = f1_score_to_y(avg_f1)
    Z = feature_columns_to_z(len(selected_feature_columns))
    
    if dataset_type == "public":
        available_points = 15
    elif dataset_type == "private":
        available_points = 20
    else:
        raise Exception("dataset type is not valid. It should be 'public' or 'private'")

    points = available_points * ((X+Y) / 2) * Z
    logging.info(f"Total {dataset_type} points is {points}.")
    
    # Save the result to a text file
    points_file_path = model_folder + f'/{dataset_type}_points.txt'
    with open(points_file_path, 'w') as file:
        file.write(f"{points}")

def accuracy_to_x(accuracy):
    # Round up to the nearest 100th
    rounded_accuracy = math.ceil(accuracy * 100) / 100

    if 0.95 <= rounded_accuracy <= 1.00:
        return 1
    elif 0.90 <= rounded_accuracy < 0.95:
        return 0.9
    elif 0.85 <= rounded_accuracy < 0.90:
        return 0.7
    elif rounded_accuracy < 0.85:
        return 0
    else:
        raise Exception("Invalid accuracy: The value should be between 0 and 1.")

def f1_score_to_y(f1_score):
    # Round up to the nearest 100th
    rounded_f1_score = math.ceil(f1_score * 100) / 100

    if 0.90 <= rounded_f1_score <= 1.00:
        return 1
    elif 0.85 <= rounded_f1_score < 0.90:
        return 0.9
    elif 0.80 <= rounded_f1_score < 0.85:
        return 0.7
    elif rounded_f1_score < 0.80:
        return 0
    else:
        raise Exception("Invalid F1 score: The value should be between 0 and 1.")

def feature_columns_to_z(num_columns):
    if 1 <= num_columns <= 53:
        return 1
    elif 54 <= num_columns <= 64:
        return 0.9
    elif 65 <= num_columns <= 80:
        return 0.7
    elif num_columns > 80:
        return 0
    else:
        raise Exception("Invalid number of features: The value should be a positive integer for the number of columns.")

if __name__ == "__main__":
    default_data_dir_path = os.path.join(os.getcwd(), "data/39-bus-measurements")
    default_model_dir_path = os.path.join(os.getcwd(), "model")
    parser = argparse.ArgumentParser(description="Test the given neural network model using the test dataset.")
    parser.add_argument("--data_folder", default=default_data_dir_path, help="Path to the folder containing test CSV files")
    parser.add_argument("--model_folder", default=default_model_dir_path, help="Path to the folder containing the model")
    parser.add_argument("--dataset_type", choices=['public', 'private'], default='public', help="Dataset type: 'public' or 'private'")
    
    args = parser.parse_args()
    main(args.data_folder, args.model_folder, args.dataset_type)