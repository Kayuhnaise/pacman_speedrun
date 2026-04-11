import os
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from boards import ATARI_STYLE_BOARD, SMALL_BOARD
from env import PacmanEnv
from state_encoder import encode_state
from dqn_model import DQNCNN
from replay_buffer import ReplayBuffer


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


def epsilon_greedy_action(q_net, state, epsilon, num_actions, device):
    if random.random() < epsilon:
        return random.randrange(num_actions)

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())


def train_step(q_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = q_net(states_t).gather(1, actions_t)

    with torch.no_grad():
        # Double DQN:
        # 1) online network chooses best next action
        next_actions = q_net(next_states_t).argmax(dim=1, keepdim=True)
        # 2) target network evaluates that action
        next_q_values = target_net(next_states_t).gather(1, next_actions)
        targets = rewards_t + gamma * next_q_values * (1.0 - dones_t)

    loss = nn.MSELoss()(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def main():
    board_name = os.environ.get("BOARD", "atari")
    seed = int(os.environ.get("SEED", "42"))
    episodes = int(os.environ.get("TRAIN_EPISODES", "300"))
    max_steps = int(os.environ.get("MAX_STEPS", "500"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    gamma = float(os.environ.get("GAMMA", "0.99"))
    lr = float(os.environ.get("LR", "0.0005"))
    buffer_capacity = int(os.environ.get("BUFFER_CAPACITY", "50000"))
    target_update_freq = int(os.environ.get("TARGET_UPDATE_FREQ", "5"))
    epsilon_start = float(os.environ.get("EPSILON_START", "1.0"))
    epsilon_end = float(os.environ.get("EPSILON_END", "0.05"))
    epsilon_decay = float(os.environ.get("EPSILON_DECAY", "0.98"))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    state = encode_state(env, obs)

    input_channels = state.shape[0]
    rows, cols = state.shape[1], state.shape[2]
    num_actions = 5  # UP, DOWN, LEFT, RIGHT, STAY
    q_net = DQNCNN(input_channels, rows, cols, num_actions).to(device)
    target_net = DQNCNN(input_channels, rows, cols, num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    output_dir = Path("outputs")
    model_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    ensure_dir(model_dir)
    ensure_dir(logs_dir)

    train_log_path = logs_dir / f"dqn_train_{board_name}.csv"

    epsilon = epsilon_start
    episode_rows = []

    for episode in range(1, episodes + 1):
        obs, info = safe_reset(env, seed=seed + episode)
        state = encode_state(env, obs)

        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = epsilon_greedy_action(q_net, state, epsilon, num_actions, device)
            next_obs, reward, terminated, truncated, info = step_unpack(env, action)
            done = bool(terminated or truncated)

            next_state = encode_state(env, next_obs)

            replay_buffer.push(state, action, reward, next_state, done)

            if steps % 4 == 0:
                loss = train_step(
                    q_net=q_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                )

                if loss is not None:
                    total_loss += loss
                    loss_count += 1

            state = next_state
            total_reward += reward
            steps += 1

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
        won = int(info["remaining_pellets"] == 0)

        row = {
            "episode": episode,
            "reward": round(total_reward, 2),
            "steps": steps,
            "deaths": info["deaths"],
            "remaining_pellets": info["remaining_pellets"],
            "won": won,
            "epsilon": round(epsilon, 4),
            "avg_loss": round(avg_loss, 6),
        }
        episode_rows.append(row)

        print(
            f"Episode {episode:03d} | reward={total_reward:.2f} | "
            f"steps={steps} | deaths={info['deaths']} | "
            f"remaining={info['remaining_pellets']} | won={won} | "
            f"epsilon={epsilon:.4f} | loss={avg_loss:.6f}"
        )

    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "reward",
                "steps",
                "deaths",
                "remaining_pellets",
                "won",
                "epsilon",
                "avg_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    model_path = model_dir / f"dqn_{board_name}.pt"
    torch.save(q_net.state_dict(), model_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved training log to: {train_log_path}")


if __name__ == "__main__":
    main()