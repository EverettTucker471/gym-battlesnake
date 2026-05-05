import torch
import random
import numpy as np
from stable_baselines3 import PPO

# Save the real torch.load function
original_load = torch.load

# Create a wrapper that removes the buggy flags
def patched_load(*args, **kwargs):
    kwargs.pop('weights_only', None)
    kwargs['mmap'] = False # Disable mmap just to be safe
    return original_load(*args, **kwargs)

# Override PyTorch's load function globally in this script
torch.load = patched_load

def linear_schedule(initial_value):
    """
    Implements a linear schedule for the learning rate,
    so it decreases over time as the model learns
    """
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


def log_schedule(initial_value):
    """
    Implements a logarithmic scheduler for the learning rate,
    so it decreases logarithmically over time as the model learns
    """
    def func(progress_remaining):
        return np.log(progress_remaining) * initial_value
    return func


from gymbattlesnake import ParallelBattlesnakeEnv
from updated_policy import UpdatedPolicy

# Random agent defined for the intial epoch
class RandomAgent:
    def predict(self, obs, deterministic=False):
        # Returns a random action (0-3) for every opponent in every environment
        batch_size = obs.shape[0]
        actions = np.random.randint(0, 4, (batch_size,))
        return actions, None
        
def main():
    training = True  # Change for evaluation vs. training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}!")

    # Initializing environment with a random agent
    n_envs = 16
    env = ParallelBattlesnakeEnv(
        n_threads=4, 
        n_envs=n_envs, 
        n_opponents = 1,
        opponent=RandomAgent(),
        device=device, 
        fixed_orientation=True,
        use_symmetry=False,
        dtype=torch.float32
    )

    # 4. Setup Model Policy
    policy_kwargs = dict(
        features_extractor_class=UpdatedPolicy,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # 5. Initialize PPO Model
    model = PPO(
        "CnnPolicy", 
        env,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        learning_rate=0.0003,
        clip_range=0.2,
        n_steps=512,
        batch_size=1024,
        n_epochs=4,
        ent_coef=0.01,
        target_kl=0.03,
        tensorboard_log="./battlesnake_tb_logs/",
    )

    # ==========================================
    # SELF-PLAY CURRICULUM TRAINING LOOP
    # ==========================================
    model_name = "ppo_battlesnake_latest.pth"
    if training:
        iterations = 15
        steps_per_iteration = 100000
        historical_weights = []  # A list of all the models that came before this one, so we can play against older opponents for stability
        print("Starting Self-Play Training...")
        for i in range(iterations):
            print(f"\n--- Self-Play Iteration {i+1}/{iterations} ---")

            model.learn(total_timesteps=steps_per_iteration, reset_num_timesteps=False, tb_log_name="PPO_SelfPlay_UpdatedCNN")

            current_weights = {k: v.cpu().clone() for k, v in model.policy.state_dict().items()}
            torch.save(current_weights, model_name)
            historical_weights.append(current_weights)

            new_opponent = PPO("CnnPolicy", env, device=device, verbose=1, policy_kwargs=policy_kwargs)

            if random.random() < 0.6:
                new_opponent.policy.load_state_dict(current_weights)
                print("Playing against the LATEST model")
            else:
                random_historical_weights = random.choice(historical_weights)
                new_opponent.policy.load_state_dict(random_historical_weights)
                print("Playing against a HISTORICAL model")

            new_opponent.policy.to(device)
            env.opponent = new_opponent

        print("Training Complete!")
    
        # Clean up GPU context before evaluation
        env.close()
        del model
        del new_opponent

    # ==========================================
    # EVALUATION
    # ==========================================
    # Re-initialize for evaluation
    eval_env = ParallelBattlesnakeEnv(
        n_threads=1, 
        n_envs=1, 
        n_opponents=1, 
        opponent=RandomAgent(), 
        device=device, 
        fixed_orientation=True, 
        use_symmetry=False
    )
    
    final_model = PPO("CnnPolicy", env, device=device, verbose=1, policy_kwargs=policy_kwargs)
    final_weights = torch.load(model_name, map_location=device, weights_only=True)
    final_model.policy.load_state_dict(final_weights)

    # Creating a clone so the final model can play against itself
    eval_opponent = PPO("CnnPolicy", eval_env, device=device, verbose=1, policy_kwargs=policy_kwargs)
    eval_opponent.policy.load_state_dict(final_weights)
    eval_env.opponent = eval_opponent

    obs = eval_env.reset()
    
    print("Evaluating...")
    for _ in range(1000):
        # Change deterministic=True to false if you see repeated behavior between evaluation iterations
        actions, _ = final_model.predict(obs, deterministic=False) 
        obs, _, _, _ = eval_env.step(actions) 
        eval_env.render()

if __name__ == "__main__":
    main()