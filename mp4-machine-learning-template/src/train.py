import torch
import argparse
from data_reader import DataReader
from power_system_nn import PowerSystemNN
from trainer import Trainer
import os
from joblib import dump

def main(data_folder, model_folder) -> None:
    """
    Main function to execute model training pipeline.

    This function orchestrates the process of training a neural network model by performing the following steps:
    1. Load and preprocess the data: This involves reading the data and applying necessary preprocessing steps to make it suitable for training the model.
    2. Initialize and train the neural network: Sets up the neural network architecture and trains it on the preprocessed data.
    3. Save the model: After training, the trained model is saved to disk for future inference.
    4. Save the scaler: The scaler used for data normalization during preprocessing is saved separately to ensure the same scaling is applied during future predictions.
    5. Save the selected feature columns: Keeps a record of the feature columns that were selected during preprocessing to ensure consistency in future data processing.

    Returns:
    None
    """
    feature_file_path = data_folder + "/train_features.csv"
    label_file_path = data_folder + "/train_labels.csv"

    # 1. Load and preprocess data
    train_data_reader = DataReader(feature_file_path, label_file_path)
    train_data_reader.load_train_data()

    # 2. Initialize and train the neural network
    model = PowerSystemNN(input_dim = train_data_reader.feature_dim, output_dim = train_data_reader.label_dim)
    trainer = Trainer(model)
    trainer.train_model(train_data_reader)

    # 3. Save the model
    torch.save(model.state_dict(), model_folder + '/model.pth')
    
    # 4. Save the scaler used to normalize data
    dump(train_data_reader.scaler, model_folder + '/scaler.joblib')

    # 5. Save the selected feature columns
    with open(model_folder + '/selected_feature_columns.txt', 'w') as file:
        for item in train_data_reader.selected_feature_columns:
            file.write("%s\n" % item)


if __name__ == "__main__":
    default_data_dir_path = os.path.join(os.getcwd(), "data/39-bus-measurements")
    default_model_dir_path = os.path.join(os.getcwd(), "model")
    parser = argparse.ArgumentParser(description="Train a neural network model using the train dataset.")
    parser.add_argument("--data_folder", default=default_data_dir_path, help="Path to the folder containing training CSV files")
    parser.add_argument("--model_folder", default=default_model_dir_path, help="Path to the folder to save the model")
    
    args = parser.parse_args()
    main(args.data_folder, args.model_folder)
