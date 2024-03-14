# linear.py - a very simple MLP model for model save/load testing
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, embed_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes),
        )
        
    def forward(self, x):
        return self.model(x)