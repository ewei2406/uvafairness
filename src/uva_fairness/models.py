import torch
from torch import nn

class model(nn.Module):
    """
    Simple model with 3 layers; hidden layer with 64 neurons. 
    Specify size of input and output tensors.
    """
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def base_GCN(input_dim=(28*28), output_dim=10):
    return model(input_dim=input_dim, output_dim=output_dim)
