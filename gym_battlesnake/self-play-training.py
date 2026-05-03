import torch
import numpy as np
from stable_baselines3 import PPO

from gymbattlesnake import ParallelBattlesnakeEnv
from updated_policy import UpdatedPolicy

# Random agent defined for the intial epoch
class RandomAgent:
    def predict(self, obs, deterministic=False):
        # Returns a random action (0-3) for every opponent in every environment
        batch_size = obs.shape[0]
        actions = torch.randint(0, 4, (batch_size,), device=obs.device)
        return actions, None
        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}!")

    # Initializing environment with a random agent
    n_envs = 16
    n_opponents = 7
    env = ParallelBattlesnakeEnv(
        n_threads=4, 
        n_envs=n_envs, 
        n_opponents=n_opponents, 
        opponent=RandomAgent(),
        device=device, 
        fixed_orientation=True,
        use_symmetry=True,
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
        learning_rate=0.00003,
        clip_range=0.1,
        n_steps=512,
        batch_size=256,
        ent_coef=0.01,
        target_kl=0.02
    )

    # ==========================================
    # SELF-PLAY CURRICULUM TRAINING LOOP
    # ==========================================
    iterations = 10 
    steps_per_iteration = 200
    model_name = "ppo_battlesnake_latest"

    print("Starting Self-Play Training...")
    for i in range(iterations):
        print(f"\n--- Self-Play Iteration {i+1}/{iterations} ---")
        
        # Train the model
        model.learn(total_timesteps=steps_per_iteration, reset_num_timesteps=False)
        
        # Save the latest snapshot
        model.save(model_name)
        
        print("Updating opponent to the latest model version...")
        # Load the frozen snapshot as a completely independent object
        new_opponent = PPO.load(model_name, env=env, device=device, print_system_info=True, force_reset=True)
        
        # Hot-swap the C++ wrapper's opponent to the newly trained model!
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
        n_threads=1, n_envs=1, n_opponents=7, opponent=RandomAgent(), 
        device=device, fixed_orientation=True, use_symmetry=True
    )
    
    final_model = PPO.load(model_name, device=device, print_system_info=True)
    obs = eval_env.reset()
    
    print("Evaluating...")
    for _ in range(1000):
        # VecEnv returns an array of actions, so we pass it directly to step
        actions, _ = final_model.predict(obs) 
        obs, _, _, _ = eval_env.step(actions) 
        eval_env.render()

if __name__ == "__main__":
    main()