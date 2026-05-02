import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomPolicy(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network feature extractor for Battlesnake.
    Replaces the old FeedForwardPolicy / tf.variable_scope setup.
    """
    def __init__(self, observation_space, features_dim=512):
        # We assume the observation_space shape is (6, 39, 39) - Channel First!
        super(CustomPolicy, self).__init__(observation_space=observation_space, features_dim=features_dim)

        n_input_channels = observation_space.shape[0]

        # 1. The Convolutional Layers
        self.cnn = nn.Sequential(
            # conv1: (N, 6, 39, 39) -> (N, 16, 18, 18)
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2),
            nn.ELU(),
            
            # conv2: (N, 16, 18, 18) -> (N, 32, 8, 8)
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(),
            
            # Flatten to 1D array for the Linear layers
            nn.Flatten(),
        )

        # Compute the flattened output size automatically
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # 2. The Fully Connected (Linear) Layers
        self.linear = nn.Sequential(
            # fc1: (N, 2048) -> (N, 1024)
            nn.Linear(n_flatten, 1024),
            nn.ELU(),
            
            # fc2: (N, 1024) -> (N, 512)
            nn.Linear(1024, features_dim),
            nn.ELU()
        )

    def forward(self, observations):
        # Pass observations through the CNN, then through the Linear layers
        return self.linear(self.cnn(observations))