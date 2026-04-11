import os
import csv
from pathlib import Path

import numpy as np
import torch

from boards import ATARI_STYLE_BOARD, SMALL_BOARD
from env import PacmanEnv
from state_encoder import encode_state
from dqn_model import DQNCNN


def safe_reset(env, seed=None):
    result = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(result, tuple):
        return result
    return result, {}


def step_unpack(env, action):
    result = env.step(action)
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, done, False, info
    obs, reward, terminated, truncated, info = result
    return obs, reward, terminated, truncated, info


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def select_board(name: str):
    return SMALL_BOARD if name.lower() == "small" else ATARI_STYLE_BOARD


def main():
    board_name = os.environ.get("BOARD", "atari")
    episodes = int(os.environ.get("EPISODES", "10"))
    max_steps = int(os.environ.get("MAX_STEPS", "200"))
    seed = int(os.environ.get("SEED", "42"))
    render = os.environ.get("RENDER", "1") not in ("0", "false", "False")

    board = select_board(board_name)
    env = PacmanEnv(
        board=board,
        max_steps=max_steps,
        max_deaths=3,
        scared_duration=20,
        ghost_respawn_steps=10,
        ghost_spawn_interval=10,
        max_ghosts=4,
        seed=seed,
    )

    obs, info = safe_reset(env, seed=seed)
    sample_state = encode_state(env, obs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = DQNCNN(sample_state.shape[0], env.rows, env.cols, 5).to(device)

    model_path = Path("outputs") / "models" / f"dqn_{board_name}.pt"
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    output_dir = Path("outputs")
    logs_dir = output_dir / "logs"
    ensure_dir(logs_dir)

    eval_log_path = logs_dir / f"dqn_eval_{board_name}.csv"
    rows = []

    rewards = []

    for ep in range(1, episodes + 1):
        obs, info = safe_reset(env, seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        if render:
            print(f"\n=== DQN Episode {ep} ===")
            env.render()

        while not done and steps < max_steps:
            state = encode_state(env, obs)
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action = int(torch.argmax(q_net(state_t), dim=1).item())

            obs, reward, terminated, truncated, info = step_unpack(env, action)
            done = bool(terminated or truncated)
            total_reward += reward
            steps += 1

            if render:
                print(f"Action: {action}")
                env.render()

        won = int(info["remaining_pellets"] == 0)

        rows.append(
            {
                "episode": ep,
                "policy": "dqn",
                "board": board_name,
                "total_reward": round(total_reward, 2),
                "steps": steps,
                "deaths": info["deaths"],
                "remaining_pellets": info["remaining_pellets"],
                "won": won,
            }
        )

        rewards.append(total_reward)
        print(
            f"Episode {ep} finished | reward={total_reward:.2f}, "
            f"steps={steps}, deaths={info['deaths']}, "
            f"remaining_pellets={info['remaining_pellets']}, won={won}"
        )

    with open(eval_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "policy",
                "board",
                "total_reward",
                "steps",
                "deaths",
                "remaining_pellets",
                "won",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAverage reward over {episodes} episodes: {np.mean(rewards):.2f}")
    print(f"Saved DQN evaluation log to: {eval_log_path}")


if __name__ == "__main__":
    main()