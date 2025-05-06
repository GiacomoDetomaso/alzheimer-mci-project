import torch
import torch.nn as nn

from typing import Optional, Sequence
from collections import OrderedDict

class CNN3DModel(nn.Module):
    def __init__(self,
			in_channels = int,
			in_size = int,
			out_channels = int,
			dropout_prob: Optional[float] = 0.3,
			hidden_sizes: Optional[Sequence[int]] = (64, 64, 128, 256),
			name: Optional[str] = 'CNN3DModel'
    ):
        
        super(CNN3DModel, self).__init__()

        self.in_channels = in_channels
        self.in_size = in_size
        self.out_channels = out_channels
        self.dropout_prob = dropout_prob
        self.name = name

        layers_features = OrderedDict()

        self.hidden_sizes = (in_channels, ) + hidden_sizes

        for i in range(0, len(self.hidden_sizes) - 1):
            layers_features[f"conv{i+1}"] = nn.Conv3d(
				in_channels = self.hidden_sizes[i],
				out_channels = self.hidden_sizes[i + 1],
				kernel_size = 3,
				padding = 1
			)

            layers_features[f"relu{i+1}"] = nn.ReLU()
            layers_features[f"pool{i+1}"] = nn.MaxPool3d(kernel_size = 2)
            layers_features[f"batchnorm{i+1}"] = nn.BatchNorm3d(num_features = self.hidden_sizes[i + 1])

        self.features = nn.Sequential(layers_features)

        # Define the classifier head
        self.classifier = nn.Sequential(OrderedDict([
			('global_avg_pool', nn.AdaptiveAvgPool3d(1)),
			('flatten', nn.Flatten()),
			('fc1', nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1] * 2)),
			('relu_fc1', nn.ReLU()),
			('dropout', nn.Dropout(p = self.dropout_prob)),
			('fc2', nn.Linear(self.hidden_sizes[-1] * 2, self.out_channels))
		]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)  # Apply feature extractor
        y = self.classifier(y)  # Apply classifier
            
        return y