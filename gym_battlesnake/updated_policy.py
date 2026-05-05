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
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_obs = torch.as_tensor(observation_space.sample()[None]).float()
            flatten_dim = self.cnn(dummy_obs).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(in_features=flatten_dim, out_features=features_dim),
            nn.LeakyReLU(),
        )

    def forward(self, observations):
        # Pass observations through the CNN, then through the Linear layers
        return self.linear(self.cnn(observations))
    