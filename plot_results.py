from pathlib import Path
import glob

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(series, window=10):
    return series.rolling(window=window, min_periods=1).mean()


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

        plt.figure(figsize=(10, 5))
        plt.plot(df["episode"], df["reward"], label="Reward")
        plt.plot(df["episode"], moving_average(df["reward"], 10), label="Reward MA(10)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"DQN Training Reward - {stem}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_reward.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(df["episode"], df["avg_loss"], label="Loss")
        plt.xlabel("Episode")
        plt.ylabel("Average Loss")
        plt.title(f"DQN Training Loss - {stem}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_loss.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(df["episode"], df["won"], label="Win")
        plt.plot(df["episode"], moving_average(df["won"], 10), label="Win Rate MA(10)")
        plt.xlabel("Episode")
        plt.ylabel("Won")
        plt.title(f"DQN Win Trend - {stem}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_winrate.png")
        plt.close()

        print(f"Saved DQN plots for {file_path}")


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

    if "policy" not in df.columns:
        print("Summary files do not contain policy column.")
        return

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

        plt.figure(figsize=(8, 5))
        plt.bar(df["policy"], df[col])
        plt.xlabel("Policy")
        plt.ylabel(title)
        plt.title(f"Policy Comparison - {title}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(out_dir / f"policy_comparison_{col}.png")
        plt.close()

    print("Saved policy comparison plots.")


if __name__ == "__main__":
    plot_dqn_training()
    plot_policy_comparison()