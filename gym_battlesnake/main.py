import numpy as np
import typing
from stable_baselines3 import PPO
import torch
from updated_policy import UpdatedPolicy
from gymbattlesnake import BattlesnakeEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_kwargs = dict(
    features_extractor_class=UpdatedPolicy,
    features_extractor_kwargs=dict(features_dim=512),
)

dummy_env = BattlesnakeEnv(n_threads=1, n_envs=1, opponents=[], fixed_orientation=True)
model = PPO("CnnPolicy", dummy_env, policy_kwargs=policy_kwargs, device=device)
weights = torch.load("ppo_battlesnake_latest.pth", map_location=device, weights_only=True)
model.policy.load_state_dict(weights)
dummy_env.close()

NUM_LAYERS = 17
LAYER_WIDTH = 23
LAYER_HEIGHT = 23

def game_state_to_obs(game_state: typing.Dict) -> np.ndarray:
    obs = np.zeros((NUM_LAYERS, LAYER_WIDTH, LAYER_HEIGHT), dtype=np.uint8)
    
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    me = game_state["you"]
    my_head = me["body"][0]
    my_length = len(me["body"])
    hx, hy = my_head["x"], my_head["y"]
    cx, cy = LAYER_WIDTH // 2, LAYER_HEIGHT // 2  # center = 11,11

    def assign(bx, by, layer, val):
        ox = (bx - hx) + cx
        oy = (by - hy) + cy
        # swap: model expects x as row, y as col
        if 0 <= ox < LAYER_WIDTH and 0 <= oy < LAYER_HEIGHT:
            obs[layer, oy, ox] = min(255, obs[layer, oy, ox] + val)

    # Layer 5: gameboard (valid tiles)
    for x in range(board_width):
        for y in range(board_height):
            assign(x, y, 5, 1)

    # Layer 4: food
    for food in game_state["board"]["food"]:
        assign(food["x"], food["y"], 4, 1)

    # Layer 6: my head mask
    assign(hx, hy, 6, 1)

    # Count alive snakes for layers 10-16
    all_snakes = game_state["board"]["snakes"]
    alive_count = len(all_snakes)
    alive_layer = max(0, min(6, alive_count - 2))
    for x in range(board_width):
        for y in range(board_height):
            assign(x, y, 10 + alive_layer, 1)

    # Process all snakes
    for snake in all_snakes:
        is_me = snake["id"] == me["id"]
        body = snake["body"]
        snake_length = len(body)
        health = snake["health"]

        # Layer 0: health on head
        assign(body[0]["x"], body[0]["y"], 0, health)

        # Layer 3: opponent head length >= me
        if not is_me:
            assign(body[0]["x"], body[0]["y"], 3, 1 if snake_length >= my_length else 0)

        # Body layers
        tail_1 = body[-1]
        tail_2 = body[-2] if len(body) > 1 else tail_1

        for i, seg in enumerate(body):
            bx, by = seg["x"], seg["y"]
            # Layer 1: all bodies
            assign(bx, by, 1, 1)
            # Layer 2: segment number (distance from tail, capped at 255)
            assign(bx, by, 2, min(i + 1, 255))

            if not is_me:
                if snake_length >= my_length:
                    # Layer 8: bigger/equal opponent bodies
                    assign(bx, by, 8, 1 + snake_length - my_length)
                else:
                    # Layer 9: smaller opponent bodies
                    assign(bx, by, 9, my_length - snake_length)

        # Layer 7: double tail (tail segment same as second-to-last)
        if tail_1["x"] == tail_2["x"] and tail_1["y"] == tail_2["y"]:
            assign(tail_1["x"], tail_1["y"], 7, 1)

    return obs

ACTION_TO_MOVE = {0: "down", 1: "up", 2: "left", 3: "right"}

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "justjack",
        "color": "#CEA2FD",
        "head": "default",
        "tail": "default",
    }

def start(game_state: typing.Dict):
    print("GAME START")

def end(game_state: typing.Dict):
    print("GAME OVER")

def move(game_state: typing.Dict) -> typing.Dict:
    obs = game_state_to_obs(game_state)
    obs_batch = obs[np.newaxis, :]
    action, _ = model.predict(obs_batch, deterministic=True)
    move_str = ACTION_TO_MOVE[int(action[0])]
    head = game_state['you']['head']
    print(f"MOVE {game_state['turn']}: action={int(action[0])} -> {move_str}, head={head}")
    return {"move": move_str}

if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})