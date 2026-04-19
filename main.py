import os
import sys
import csv
from pathlib import Path
from typing import Callable, Optional

import imageio.v2 as imageio

from boards import ATARI_STYLE_BOARD, PAC_BOARD
from env import PacmanEnv
import policies


def safe_reset(env, seed: Optional[int] = None):
    result = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(result, tuple):
        return result
    return result, {}


def step_unpack(env, action):
    result = env.step(action)
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, done, False, info
    else:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated, truncated, info


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_gif(frames, out_path: Path, fps: int = 4):
    if not frames:
        return
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(out_path, frames, duration=duration)


def get_board(board_name: str):
    board_name = (board_name or "atari").lower()
    if board_name == "atari":
        return ATARI_STYLE_BOARD
    if board_name == "pac":
        return PAC_BOARD
    return ATARI_STYLE_BOARD


def run(
    policy: Callable,
    policy_name: str,
    board_name: str = "atari",
    episodes: int = 3,
    max_steps: int = 500,
    render: bool = True,
    seed: Optional[int] = None,
    save_videos: bool = True,
    fps: int = 4,
):
    board = get_board(board_name)

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

    output_dir = Path("outputs")
    videos_dir = output_dir / "videos"
    logs_dir = output_dir / "logs"
    ensure_dir(output_dir)
    ensure_dir(videos_dir)
    ensure_dir(logs_dir)

    episode_log_path = logs_dir / f"{policy_name}_{board_name}_episode_log.csv"
    summary_log_path = logs_dir / f"{policy_name}_{board_name}_summary.csv"

    episode_rows = []

    for index in range(1, episodes + 1):
        episode_seed = seed + index if seed is not None else None
        obs, info = safe_reset(env, seed=episode_seed)
        total_reward = 0.0
        done = False
        steps = 0
        frames = []

        print(f"\n=== Episode {index} | board={board_name} | policy={policy_name} ===")
        if render:
            env.render()

        if save_videos and hasattr(env, "render_frame"):
            frames.append(env.render_frame())

        while not done and steps < max_steps:
            action = policy(env, obs, info, steps)
            obs, reward, terminated, truncated, info = step_unpack(env, action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1

            if render:
                print(f"Action: {action}")
                env.render()

            if save_videos and hasattr(env, "render_frame"):
                frames.append(env.render_frame())

        won = int(info["remaining_pellets"] == 0)

        row = {
            "episode": index,
            "policy": policy_name,
            "board": board_name,
            "total_reward": round(total_reward, 2),
            "steps": steps,
            "deaths": info["deaths"],
            "remaining_pellets": info["remaining_pellets"],
            "won": won,
            "seed": episode_seed,
        }
        episode_rows.append(row)

        if save_videos and frames:
            gif_path = videos_dir / f"{policy_name}_{board_name}_episode_{index}.gif"
            save_gif(frames, gif_path, fps=fps)
            print(f"Saved video: {gif_path}")

        print(
            f"Episode {index} finished | "
            f"reward={total_reward:.2f}, "
            f"steps={steps}, "
            f"deaths={info['deaths']}, "
            f"remaining_pellets={info['remaining_pellets']}, "
            f"won={won}"
        )

    with open(episode_log_path, "w", newline="", encoding="utf-8") as f:
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
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    avg_reward = sum(r["total_reward"] for r in episode_rows) / len(episode_rows)
    avg_steps = sum(r["steps"] for r in episode_rows) / len(episode_rows)
    avg_deaths = sum(r["deaths"] for r in episode_rows) / len(episode_rows)
    avg_remaining = sum(r["remaining_pellets"] for r in episode_rows) / len(episode_rows)
    win_rate = sum(r["won"] for r in episode_rows) / len(episode_rows)

    with open(summary_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "board",
                "episodes",
                "avg_reward",
                "avg_steps",
                "avg_deaths",
                "avg_remaining_pellets",
                "win_rate",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "policy": policy_name,
                "board": board_name,
                "episodes": len(episode_rows),
                "avg_reward": round(avg_reward, 2),
                "avg_steps": round(avg_steps, 2),
                "avg_deaths": round(avg_deaths, 2),
                "avg_remaining_pellets": round(avg_remaining, 2),
                "win_rate": round(win_rate, 3),
            }
        )

    print(f"\nSaved episode log: {episode_log_path}")
    print(f"Saved summary log: {summary_log_path}")


if __name__ == "__main__":
    episodes = int(os.environ.get("EPISODES", "5"))
    render = os.environ.get("RENDER", "1") not in ("0", "false", "False")
    max_steps = int(os.environ.get("MAX_STEPS", "500"))
    seed_env = os.environ.get("SEED", "42")
    seed = int(seed_env) if seed_env is not None else None
    policy_name = os.environ.get("POLICY", "bfs_policy")
    board_name = os.environ.get("BOARD", "atari")
    save_videos = os.environ.get("SAVE_VIDEOS", "1") not in ("0", "false", "False")
    fps = int(os.environ.get("FPS", "4"))

    policy = policies.get_attribute(policy_name)

    if not callable(policy):
        print(f"Error: policy '{policy_name}' is not callable.")
        sys.exit(1)

    print(
        f"Starting Pacman for {episodes} episodes "
        f"using policy={policy_name}, board={board_name}"
    )

    run(
        policy=policy,
        policy_name=policy_name,
        board_name=board_name,
        episodes=episodes,
        max_steps=max_steps,
        render=render,
        seed=seed,
        save_videos=save_videos,
        fps=fps,
    )