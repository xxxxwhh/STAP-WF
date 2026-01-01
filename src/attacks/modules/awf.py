import torch
from torch import nn
import numpy as np

class AWFNet(nn.Module):
    """
    AWF (Automated Website Fingerprinting) model.
    
    Input format: [batch, 1, 3000]
        - Single channel direction sequence of length 3000
    """
    def __init__(self, num_classes: int = 100):
        super(AWFNet, self).__init__()
        self.num_classes = num_classes
        
        # Define the feature extraction part of the network using a sequential container
        self.feature_extraction = nn.Sequential(
            nn.Dropout(p=0.25),  # Dropout layer with a 25% dropout rate for regularization

            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5,
                      stride=1, padding='valid', bias=False),  # First convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool1d(kernel_size=4, padding=0),  # First max pooling layer

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                      stride=1, padding='valid', bias=False),  # Second convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool1d(kernel_size=4, padding=0),  # Second max pooling layer

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                      stride=1, padding='valid', bias=False),  # Third convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool1d(kernel_size=4, padding=0),  # Third max pooling layer
        )
        
        # Define the classifier part of the network
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the tensor to a vector
            nn.Linear(in_features=32*45, out_features=num_classes)  # Fully connected layer for classification
        )

    def forward(self, x):
        # Ensure the input tensor has the expected shape
        assert x.shape[-1] == 3000, f"Expected input with 3000 elements, got {x.shape[-1]}"
        
        # Pass the input through the feature extraction part
        x = self.feature_extraction(x)
        
        # Pass the output through the classifier
        x = self.classifier(x)
        
        return x
