import torch

from gymbattlesnake import BattlesnakeEnv
from custom_policy import CustomPolicy
# from ..gym_battlesnake.gymbattlesnake import BattlesnakeEnv
# from ..gym_battlesnake.custom_policy import CustomPolicy
from stable_baselines3 import PPO

# Grabbing GPU device
device = torch.rand(1).device
print(device)
env = BattlesnakeEnv(n_threads=4, n_envs=16, device=device)
model_name = "ppo_trained_model"
policy_kwargs = dict(
    features_extractor_class=CustomPolicy,
    features_extractor_kwargs=dict(features_dim=128),
)

def train():
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, device=device, verbose=1)
    model.learn(total_timesteps=100000)
    model.save(model_name)


def load():
    model = PPO.load(model_name, device=device)

    obs = env.reset()
    for _ in range(10000):
        action,_ = model.predict(obs)
        obs,_,_,_,_ = env.step(action)
        env.render()

train()
load()
