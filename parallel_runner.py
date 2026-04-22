"""
Parallel episode runner using multiprocessing.

Splits episodes across multiple processes for faster execution.
Each process runs a subset of episodes with proper seed management,
and all results are consolidated into unified logs.
"""

import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path

from env import PacmanEnv
from main import _run_single_episode, ensure_dir, get_board, save_gif
import policies


def get_optimal_num_processes(episodes: int) -> tuple[int, str]:
    """Determine process count based on CPU availability and episode count."""
    cpu_count = os.cpu_count() or 1
    max_useful = max(1, cpu_count - 1)
    optimal = min(max_useful, max(1, episodes))
    reason = (
        f"{cpu_count} CPUs, {optimal} process threads"
    )
    return optimal, reason


def worker_run_episodes(
    policy_name: str,
    board_name: str,
    episode_indices: list[int],
    base_seed: int,
    max_steps: int,
    save_videos: bool,
) -> list[dict]:
    """Run a chunk of episodes inside one worker process."""
    policy = policies.get_attribute(policy_name)
    if not callable(policy):
        raise RuntimeError(f"Policy '{policy_name}' is not callable in subprocess")

    board = get_board(board_name)
    env = PacmanEnv(
        board=board,
        max_steps=max_steps,
        max_deaths=3,
        scared_duration=20,
        ghost_respawn_steps=10,
        ghost_spawn_interval=10,
        max_ghosts=4,
        seed=None,
    )

    episode_rows: list[dict] = []
    for episode_index in episode_indices:
        episode_seed = base_seed + episode_index if base_seed is not None else None
        row, frames = _run_single_episode(
            policy=policy,
            env=env,
            policy_name=policy_name,
            board_name=board_name,
            episode_index=episode_index,
            episode_seed=episode_seed,
            max_steps=max_steps,
            render=False,
            save_videos=save_videos,
        )

        row["_frames"] = frames if save_videos else []
        episode_rows.append(row)

    return episode_rows


def run_parallel(
    policy_name: str,
    board_name: str = "atari",
    episodes: int = 3,
    max_steps: int = 500,
    seed: Optional[int] = None,
    save_videos: bool = True,
    fps: int = 4,
):
    """Run episodes in parallel and consolidate outputs into standard logs."""
    actual_processes, optimization_reason = get_optimal_num_processes(episodes)

    output_dir = Path("outputs")
    videos_dir = output_dir / "videos"
    logs_dir = output_dir / "logs"
    ensure_dir(output_dir)
    ensure_dir(videos_dir)
    ensure_dir(logs_dir)

    episode_log_path = logs_dir / f"{policy_name}_{board_name}_episode_log.csv"
    summary_log_path = logs_dir / f"{policy_name}_{board_name}_summary.csv"

    print(f"Starting parallel run: {episodes} episodes")
    print(f"  {optimization_reason}")

    chunk_size = max(1, episodes // actual_processes)
    episode_chunks: list[list[int]] = []
    for i in range(actual_processes):
        start_idx = 1 + i * chunk_size
        end_idx = episodes + 1 if i == actual_processes - 1 else start_idx + chunk_size
        if start_idx < episodes + 1:
            episode_chunks.append(list(range(start_idx, end_idx)))

    all_episode_rows: list[dict] = []
    with mp.Pool(processes=actual_processes) as pool:
        results = [
            pool.apply_async(
                worker_run_episodes,
                (
                    policy_name,
                    board_name,
                    chunk,
                    seed,
                    max_steps,
                    save_videos,
                ),
            )
            for chunk in episode_chunks
        ]

        for i, result in enumerate(results):
            try:
                episode_rows = result.get()
                all_episode_rows.extend(episode_rows)
                print(f"Process {i} completed: {len(episode_rows)} episodes")
            except Exception as exc:
                print(f"Error in process {i}: {exc}", file=sys.stderr)
                raise

    all_episode_rows.sort(key=lambda x: x["episode"])

    if save_videos:
        for row in all_episode_rows:
            if row.get("_frames"):
                gif_path = videos_dir / f"{policy_name}_{board_name}_episode_{row['episode']}.gif"
                save_gif(row["_frames"], gif_path, fps=fps)
                print(f"Saved video: {gif_path}")
            row.pop("_frames", None)

    fieldnames = [
        "episode",
        "policy",
        "board",
        "total_reward",
        "steps",
        "deaths",
        "remaining_pellets",
        "won",
        "seed",
    ]

    # Keep _frames for video generation, but strip transport-only keys for CSV output.
    csv_rows = [{k: row.get(k) for k in fieldnames} for row in all_episode_rows]

    with episode_log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    avg_reward = sum(r["total_reward"] for r in all_episode_rows) / len(all_episode_rows)
    avg_steps = sum(r["steps"] for r in all_episode_rows) / len(all_episode_rows)
    avg_deaths = sum(r["deaths"] for r in all_episode_rows) / len(all_episode_rows)
    avg_remaining = sum(r["remaining_pellets"] for r in all_episode_rows) / len(
        all_episode_rows
    )
    win_rate = sum(r["won"] for r in all_episode_rows) / len(all_episode_rows)

    summary_log_path = logs_dir / f"{policy_name}_{board_name}_summary.csv"
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
                "episodes": len(all_episode_rows),
                "avg_reward": round(avg_reward, 2),
                "avg_steps": round(avg_steps, 2),
                "avg_deaths": round(avg_deaths, 2),
                "avg_remaining_pellets": round(avg_remaining, 2),
                "win_rate": round(win_rate, 3),
            }
        )

    print(f"\nSaved episode log: {episode_log_path}")
    print(f"Saved summary log: {summary_log_path}")
    print(
        f"Summary: {len(all_episode_rows)} episodes, "
        f"avg_reward={round(avg_reward, 2)}, "
        f"win_rate={round(win_rate, 3)}"
    )


if __name__ == "__main__":
    episodes = int(os.environ.get("EPISODES", "5"))
    max_steps = int(os.environ.get("MAX_STEPS", "500"))
    seed_env = os.environ.get("SEED", "0")
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
        f"Starting Pacman runner for {episodes} episodes "
        f"using policy={policy_name}, board={board_name}"
    )

    run_parallel(
        policy_name=policy_name,
        board_name=board_name,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        save_videos=save_videos,
        fps=fps,
    )