from pathlib import Path
import glob

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(series, window=10):
    return series.rolling(window=window, min_periods=1).mean()


def clean_name(name: str) -> str:
    name = str(name).replace("_policy", "")
    name = name.replace("_", " ")
    return name.title()


def plot_dqn_training():
    files = glob.glob("outputs/logs/dqn_train_*.csv")
    if not files:
        print("No DQN training logs found.")
        return

    out_dir = Path("outputs") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        df = pd.read_csv(file_path)
        stem = Path(file_path).stem

        # Reward plot
        plt.figure(figsize=(9, 5))
        plt.plot(df["episode"], df["reward"], linewidth=1.5, label="Episode Reward")
        plt.plot(
            df["episode"],
            moving_average(df["reward"], 10),
            linewidth=2.5,
            label="Reward MA(10)",
        )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN Training Reward")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_reward.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Remaining pellets plot
        plt.figure(figsize=(9,5))
        plt.plot(df["episode"], df["remaining_pellets"],linewidth=1.5, label="Remaining Pellets")
        plt.plot(df["episode"], moving_average(df["remaining_pellets"], 10), linewidth=2.5, label="Remaining Pellets MA(10)")
        plt.xlabel("Episode")
        plt.ylabel("Remaining Pellets")
        plt.title("DQN Remaining Pellets Trend")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_pellets.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Loss plot
        plt.figure(figsize=(9, 5))
        plt.plot(df["episode"], df["avg_loss"], linewidth=1.5, label="Average Loss")
        plt.plot(
            df["episode"],
            moving_average(df["avg_loss"], 10),
            linewidth=2.5,
            label="Loss MA(10)",
        )
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("DQN Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Win-rate plot
        plt.figure(figsize=(9, 5))
        plt.plot(df["episode"], df["won"], linewidth=1.2, label="Win")
        plt.plot(
            df["episode"],
            moving_average(df["won"], 10),
            linewidth=2.5,
            label="Win Rate MA(10)",
        )
        plt.xlabel("Episode")
        plt.ylabel("Win")
        plt.title("DQN Win Trend")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_winrate.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved DQN plots for {file_path}")


def build_dqn_summary():
    files = glob.glob("outputs/logs/dqn_eval_*.csv")
    if not files:
        return None

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    summary = (
        df.groupby(["policy", "board"], as_index=False)
        .agg(
            avg_reward=("total_reward", "mean"),
            avg_steps=("steps", "mean"),
            avg_deaths=("deaths", "mean"),
            avg_remaining_pellets=("remaining_pellets", "mean"),
            win_rate=("won", "mean"),
            episodes=("episode", "count"),
        )
    )

    return summary


def plot_policy_comparison():
    files = glob.glob("outputs/logs/*summary.csv")
    if not files:
        print("No summary CSV files found.")
        return

    df_list = []
    for file_path in files:
        df = pd.read_csv(file_path)
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    # Add DQN evaluation summary if available
    dqn_summary = build_dqn_summary()
    if dqn_summary is not None:
        df = pd.concat([df, dqn_summary], ignore_index=True)

    if "policy" not in df.columns:
        print("Summary files do not contain policy column.")
        return

    # Keep only Atari if present
    if "board" in df.columns and "atari" in set(df["board"].astype(str).str.lower()):
        df = df[df["board"].astype(str).str.lower() == "atari"].copy()

    # Deduplicate by taking the latest occurrence per policy
    df = df.drop_duplicates(subset=["policy"], keep="last").copy()

    df["policy_label"] = df["policy"].apply(clean_name)

    order = ["Random", "Greedy", "Bfs", "Smart Bfs", "Dqn"]
    df["sort_key"] = df["policy_label"].apply(
        lambda x: order.index(x) if x in order else 999
    )
    df = df.sort_values("sort_key")

    out_dir = Path("outputs") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("avg_reward", "Average Reward"),
        ("avg_steps", "Average Steps"),
        ("avg_deaths", "Average Deaths"),
        ("win_rate", "Win Rate"),
    ]

    for col, title in metrics:
        if col not in df.columns:
            continue

        plt.figure(figsize=(8.5, 5.2))
        bars = plt.bar(df["policy_label"], df[col])

        # value labels
        for bar, val in zip(bars, df[col]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.xlabel("Policy")
        plt.ylabel(title)
        plt.title(f"Policy Comparison: {title}")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            out_dir / f"policy_comparison_{col}_clean.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print("Saved cleaned policy comparison plots.")


if __name__ == "__main__":
    plot_dqn_training()
    plot_policy_comparison()