from flask import Flask, request, jsonify
import main as ml_snake
from stable_baselines3 import PPO
from updated_policy import UpdatedPolicy
from gymbattlesnake import BattlesnakeEnv
import torch
import numpy as np

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_kwargs = dict(
    features_extractor_class=UpdatedPolicy,
    features_extractor_kwargs=dict(features_dim=512),
)

dummy_env = BattlesnakeEnv(n_threads=1, n_envs=1, opponents=[], fixed_orientation=True)
model1 = PPO("CnnPolicy", dummy_env, policy_kwargs=policy_kwargs, device=device)
weights1 = torch.load("ppo_battlesnake_latest.pth", map_location=device, weights_only=True)
model1.policy.load_state_dict(weights1)

model2 = PPO("CnnPolicy", dummy_env, policy_kwargs=policy_kwargs, device=device)
weights2 = torch.load("ppo_battlesnake_latest.pth", map_location=device, weights_only=True)
model2.policy.load_state_dict(weights2)

dummy_env.close()

def get_safe_moves(game_state):
    head = game_state['you']['head']
    hx, hy = head['x'], head['y']
    bw = game_state['board']['width']
    bh = game_state['board']['height']
    bodies = set()
    for snake in game_state['board']['snakes']:
        for seg in snake['body'][:-1]:
            bodies.add((seg['x'], seg['y']))
    candidates = {"up": (hx, hy+1), "down": (hx, hy-1), "left": (hx-1, hy), "right": (hx+1, hy)}
    return {m: pos for m, pos in candidates.items()
            if 0 <= pos[0] < bw and 0 <= pos[1] < bh and pos not in bodies}

# --- SNAKE 1 ---
@app.route("/snake1", methods=["GET"])
def snake1_root():
    return jsonify({"apiversion": "1", "author": "justjack", "color": "#CEA2FD", "head": "default", "tail": "default"})

@app.route("/snake1/start", methods=["POST"])
def snake1_start():
    return "ok"

@app.route("/snake1/move", methods=["POST"])
def snake1_move():
    game_state = request.get_json()
    obs = ml_snake.game_state_to_obs(game_state)
    obs_batch = obs[np.newaxis, :]
    action, _ = model1.predict(obs_batch, deterministic=True)
    move_str = ml_snake.ACTION_TO_MOVE[int(action[0])]
    safe = get_safe_moves(game_state)
    if move_str not in safe and safe:
        cx = game_state['board']['width'] // 2
        cy = game_state['board']['height'] // 2
        move_str = min(safe, key=lambda m: abs(safe[m][0]-cx) + abs(safe[m][1]-cy))
    print(f"Snake1 MOVE {game_state['turn']}: action={int(action[0])} -> {move_str}, head={game_state['you']['head']}")
    return jsonify({"move": move_str})

@app.route("/snake1/end", methods=["POST"])
def snake1_end():
    return "ok"

# --- SNAKE 2 ---
@app.route("/snake2", methods=["GET"])
def snake2_root():
    return jsonify({"apiversion": "1", "author": "justjack", "color": "#FF5733", "head": "default", "tail": "default"})

@app.route("/snake2/start", methods=["POST"])
def snake2_start():
    return "ok"

@app.route("/snake2/move", methods=["POST"])
def snake2_move():
    game_state = request.get_json()
    obs = ml_snake.game_state_to_obs(game_state)
    obs_batch = obs[np.newaxis, :]
    action, _ = model2.predict(obs_batch, deterministic=True)
    move_str = ml_snake.ACTION_TO_MOVE[int(action[0])]
    safe = get_safe_moves(game_state)
    if move_str not in safe and safe:
        cx = game_state['board']['width'] // 2
        cy = game_state['board']['height'] // 2
        move_str = min(safe, key=lambda m: abs(safe[m][0]-cx) + abs(safe[m][1]-cy))
    print(f"Snake1 MOVE {game_state['turn']}: action={int(action[0])} -> {move_str}, head={game_state['you']['head']}")
    return jsonify({"move": move_str})

@app.route("/snake2/end", methods=["POST"])
def snake2_end():
    return "ok"

if __name__ == "__main__":
    app.run(port=8000, threaded=True)