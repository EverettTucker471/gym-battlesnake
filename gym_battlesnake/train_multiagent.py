import torch
import time

from gymbattlesnake import BattlesnakeEnv, ParallelBattlesnakeEnv
from updated_policy import UpdatedPolicy
from stable_baselines3 import PPO

# Grabbing GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}!")

n_opponents = 7
opponent_environment = BattlesnakeEnv(n_threads=4, 
                                        n_envs=16, 
                                        device=device, 
                                        fixed_orientation=True, 
                                        use_symmetry=True
                                    )

env = ParallelBattlesnakeEnv(n_threads=4, 
                             n_envs=16, 
                             n_opponents=n_opponents, 
                             opponent=opponent_environment,
                             device=device, 
                             fixed_orientation=True,
                             use_symmetry=True,
                             dtype=torch.float32)

model_name = "ppo_trained_model.zip"
policy_kwargs = dict(
    features_extractor_class=UpdatedPolicy,
    features_extractor_kwargs=dict(features_dim=512),
)

def evaluate():
    model = PPO("CnnPolicy", env,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=1,
                learning_rate=0.00003,
                clip_range=0.1,
                n_steps=512,
                batch_size=256,
                ent_coef=0.01,
                target_kl=0.02)
    
    model.learn(total_timesteps=2_000_000)

    obs = env.reset()
    for _ in range(10000):
        action,_ = model.predict(obs)
        obs,_,_,_ = env.step(action)
        env.render()

evaluate()
