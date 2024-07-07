from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from data_reader import DataReader
import logging
from typing import Tuple
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

class Evaluator:
    """
    A class to evaluate the performance of the neural network model using accuracy, precision, and recall metrics.

    This class is designed to assess how well a trained model performs on a dataset, typically a test dataset, provided through a DataReader instance.
    It calculates and prints out the accuracy, precision, and recall for each power system branch, as well as average metrics across all branches.

    Methods:
        evaluate(data_reader: DataReader) -> None: Evaluates the model on the provided dataset and prints performance metrics.
    """
    def __init__(self, model: nn.Module) -> None:
        """
        Initializes the Evaluator with a neural network model.
        """
        self.model: nn.Module = model # The trained neural network model to be evaluated.

    def evaluate(self, data_reader: DataReader) -> Tuple[float, float]:
        """
        Evaluates the model's performance on the dataset provided by a DataReader instance.

        The method sets the model to evaluation mode, predicts overload status on the dataset without computing gradients, and calculates accuracy, precision, and recall for each power system branch. 

        Parameters:
            data_reader (DataReader): An instance of DataReader containing the dataset on which the model is to be evaluated.

        Returns:
            avg_accuracy, avg_f1: Average accuracy and F1 score across all branches
        """
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            y_pred_tensor = self.model(data_reader.X_tensor)
            y_pred = (y_pred_tensor > 0.5).float().numpy()

            # Iterate over each branch and calculate metrics
            for i in range(data_reader.labels_df.shape[1]):
                true_labels = data_reader.labels_df.iloc[:, i]
                predicted_labels = y_pred[:, i]

                # Calculate metrics for each branch
                accuracy = accuracy_score(true_labels, predicted_labels)
                precision = precision_score(true_labels, predicted_labels, zero_division=0)
                recall = recall_score(true_labels, predicted_labels, zero_division=0)
                f1 = f1_score(true_labels, predicted_labels, zero_division=0)

                # Print metrics for each branch
                logging.info(f"Branch {i+1}:")
                logging.info(f"\tAccuracy: {accuracy}")
                logging.info(f"\tPrecision: {precision}")
                logging.info(f"\tRecall: {recall}")
                logging.info(f"\tF1 Score: {f1}")

            # Optional: Calculate and print average metrics across all branches
            avg_accuracy = sum([accuracy_score(data_reader.labels_df.iloc[:, i], y_pred[:, i]) for i in range(data_reader.labels_df.shape[1])]) / data_reader.labels_df.shape[1]
            avg_precision = sum([precision_score(data_reader.labels_df.iloc[:, i], y_pred[:, i], zero_division=0) for i in range(data_reader.labels_df.shape[1])]) / data_reader.labels_df.shape[1]
            avg_recall = sum([recall_score(data_reader.labels_df.iloc[:, i], y_pred[:, i], zero_division=0) for i in range(data_reader.labels_df.shape[1])]) / data_reader.labels_df.shape[1]
            avg_f1 = sum([f1_score(data_reader.labels_df.iloc[:, i].values, y_pred[:, i], zero_division=0) for i in range(data_reader.labels_df.shape[1])]) / data_reader.labels_df.shape[1]

            logging.info("Average Metrics Across All Branches:")
            logging.info(f"\tAverage Accuracy: {avg_accuracy}")
            logging.info(f"\tAverage Precision: {avg_precision}")
            logging.info(f"\tAverage Recall: {avg_recall}")
            logging.info(f"\tAverage F1 Score: {avg_f1}")

        return avg_accuracy, avg_f1