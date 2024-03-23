from sklearn.metrics import accuracy_score, precision_score, f1_score
import torch
import torch.nn as nn
from data_reader import DataReader
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

class Evaluator:
    """
    Class to evaluate the performance of the neural network model.
    """
    def __init__(self, model: nn.Module) -> None:
        """
        Initializes the Evaluator with a neural network model.
        """
        self.model = model  # The trained neural network model to be evaluated.

    def evaluate(self, data_reader: DataReader) -> None:
        """
        Evaluates the model's performance on the dataset provided by a DataReader instance.
        
        Your task is to calculate and print the accuracy, precision, and recall for each power system branch and
        average metrics across all branches. You can use sklearn.metrics functions for this purpose.
        """
        # Set the model to evaluation mode
        self.model.eval()
        
        # Implement prediction with the model here. Use torch.no_grad():
        
        # Convert predictions to binary format suitable for evaluation metrics
        
        # TODO: Implement the loop to calculate and print metrics for each branch
        
        # Hint: Use accuracy_score, precision_score, and recall_score from sklearn.metrics
        # Remember to handle zero division in precision and recall calculations
        
        # Optional: Calculate and print average metrics across all branches
        
        # Example placeholder for one branch (you need to implement looping and averaging)
        # true_labels = None  # You need to extract true labels for each branch from data_reader
        # predicted_labels = None  # You need to extract predicted labels for each branch
        
        # Example function calls (you need to replace None with actual variables)
        # accuracy = accuracy_score(true_labels, predicted_labels)
        # precision = precision_score(true_labels, predicted_labels, zero_division=0)
        # recall = recall_score(true_labels, predicted_labels, zero_division=0)
        
        # Log the metrics for each branch and overall averages
