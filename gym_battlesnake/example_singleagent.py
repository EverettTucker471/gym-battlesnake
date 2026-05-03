import torch
import time

from gymbattlesnake import BattlesnakeEnv
from custom_policy import CustomPolicy
from updated_policy import UpdatedPolicy
# from ..gym_battlesnake.gymbattlesnake import BattlesnakeEnv
# from ..gym_battlesnake.custom_policy import CustomPolicy
from stable_baselines3 import PPO

# Grabbing GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}!")

env = BattlesnakeEnv(n_threads=4, n_envs=16, device=device)
model_name = "ppo_trained_model.zip"
policy_kwargs = dict(
    features_extractor_class=UpdatedPolicy,
    features_extractor_kwargs=dict(features_dim=512),
)

def evaluate():
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, device=device, verbose=1)
    model.learn(total_timesteps=100000)

    model.save(model_name)

    obs = env.reset()
    for _ in range(10000):
        action,_ = model.predict(obs)
        obs,_,_,_ = env.step(action)
        env.render()

evaluate()
