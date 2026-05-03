import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class UpdatedPolicy(BaseFeaturesExtractor):
    """
    Updated feature extractor based on the medium Article:
    https://medium.com/asymptoticlabs/battlesnake-post-mortem-a5917f9a3428
    """
    def __init__(self, observation_space, features_dim=512):
        super(UpdatedPolicy, self).__init__(observation_space=observation_space, features_dim=features_dim)

        n_input_channels = observation_space.shape[0]
        assert n_input_channels ==  17

        # 1. The Convolutional Layers
        self.base = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3),
            nn.ELU(),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 128, 3),
            nn.ELU(),
            nn.Conv2d(128, 256, 3),
            nn.Flatten(),
            nn.Linear(in_features=225*256, out_features=512),
            nn.LeakyReLU(),
        )

        # Neural Network to approximate the value of the observation
        self.value_head = nn.Linear(in_features=512, out_features=1)

        # Informs probabilities of the next action based on the observation
        self.policy_head = nn.Linear(in_features=512, out_features=4)

    def forward(self, observations):
        # Pass observations through the CNN, then through the Linear layers
        return self.base(observations)