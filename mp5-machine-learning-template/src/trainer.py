import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_reader import DataReader

class Trainer:
    """
    Class responsible for training a neural network model.
    """
    def __init__(self, model: nn.Module) -> None:
        """
        Initializes the Trainer with a model and an optional learning rate.
        
        Parameters:
            model (nn.Module): The neural network model to be trained.
        """
        self.model = model
        # Define the loss function. 
        self.criterion = None  # Define the appropriate loss function
        # Define the optimizer.
        self.optimizer = None  # Initialize the optimizer with model parameters and learning rate
        
    def train_model(self, data_reader: DataReader) -> None:
        """
        Trains the model on data provided by the DataReader instance.
        
        Parameters:
            data_reader: An instance of DataReader containing the training data and labels.
        Returns:
            None
        """
        # Create DataLoader for mini-batch processing
        # Example: train_dataset = TensorDataset(data_reader.X_tensor, data_reader.y_tensor)
        # Example: train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
        epochs: int = None # Defien the number of epochs to train the model for
    
        # Training loop
        for epoch in range(epochs):
            # Iterate over batches of data
            for batch_idx, (data, target) in enumerate(train_loader):  # Use your DataLoader here
                pass
                # Reset gradients via zero_grad()
                # Forward pass
                # Compute loss
                # Backward pass and optimize via backward() and optimizer.step()
            # You can print the loss here to see how it decreases