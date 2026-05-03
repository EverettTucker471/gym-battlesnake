import torch
from gymbattlesnake import BattlesnakeEnv
from updated_policy import UpdatedPolicy
from stable_baselines3 import PPO

policy_kwargs = dict(
    features_extractor_class=UpdatedPolicy,
    features_extractor_kwargs=dict(features_dim=512),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}!")

num_agents = 4
# placeholder_env necessary for model to recognize,
# the observation and action space, and the vectorized environment
placeholder_env = BattlesnakeEnv(n_threads=4, n_envs=16)
models = [PPO("CnnPolicy", placeholder_env,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=1,
                learning_rate=0.00003,
                clip_range=0.1,
                n_steps=512,
                batch_size=256,
                ent_coef=0.01,
                target_kl=0.02) for _ in range(num_agents)]
# Close environment to free allocated resources
placeholder_env.close()

for _ in range(10):
    for model in models:
        env = BattlesnakeEnv(n_threads=4, n_envs=16, opponents=[ m for m in models if m is not model])
        model.set_env(env)
        model.learn(total_timesteps=2000000)
        env.close()

model = models[0]
env = BattlesnakeEnv(n_threads=1, n_envs=1, opponents=[ m for m in models if m is not model])
obs = env.reset()
for _ in range(10000):
    action,_ = model.predict(obs)
    obs,_,_,_ = env.step(action)
    env.render()