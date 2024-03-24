import torch
from data_reader import DataReader
from power_system_nn import PowerSystemNN
import os
from evaluator import Evaluator
from joblib import load

# Configuration
DATA_DIR_PATH = os.path.join(os.getcwd(), "data/39-bus-measurements/")
MODEL_DIR_PATH = os.path.join(os.getcwd(), "model/")
FEATURE_FILE_PATH = DATA_DIR_PATH + "test_public_features.csv"
LABEL_FILE_PATH = DATA_DIR_PATH + "test_public_labels.csv"
SCALER_FILE_PATH = MODEL_DIR_PATH + 'scaler.joblib'
MODEL_FILE_PATH = MODEL_DIR_PATH + 'model.pth'
SELECTED_FEATURES_FILE_PATH = MODEL_DIR_PATH + 'selected_feature_columns.txt'


def main() -> None:
    """
    Main function to evaluate the neural network model on a test dataset.

    This function performs the following steps:
    1. Load and preprocess test data: Utilizes the DataReader class to load test features and labels. It also loads a pre-trained scaler and a list of selected features to preprocess the data correctly.
    2. Load the model: Initializes a PowerSystemNN model with dimensions derived from the preprocessed data, then loads the model's state from a saved state dictionary.
    3. Evaluate the model: Uses the Evaluator class to assess the model's performance on the test data.

    The paths to the data, model, scaler, and selected features are configured at the start of the script.

    Returns:
        None
    """
    # 1. Load and preprocess data
    test_data_reader = DataReader(FEATURE_FILE_PATH, LABEL_FILE_PATH)
    # load the scaler
    scaler = load(SCALER_FILE_PATH)
    # load the selected features
    with open(SELECTED_FEATURES_FILE_PATH, 'r') as file:
        selected_feature_columns = file.read().splitlines()
    test_data_reader.load_test_data(scaler, selected_feature_columns)

    # 2. Load the model state
    model = PowerSystemNN(input_dim = test_data_reader.feature_dim, output_dim = test_data_reader.label_dim)
    model.load_state_dict(torch.load(MODEL_FILE_PATH))

    # 3. Evaluate the model
    evaluator = Evaluator(model)
    evaluator.evaluate(test_data_reader)


if __name__ == "__main__":
    main()