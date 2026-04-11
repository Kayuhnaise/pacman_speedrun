import pandas as pd
import glob

# Load all summary files
files = glob.glob("outputs/logs/*summary.csv")

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

print("\n=== POLICY COMPARISON ===")
print(df_all)

# Rank by reward
df_sorted = df_all.sort_values(by="avg_reward", ascending=False)

print("\n=== RANKED BY PERFORMANCE ===")
print(df_sorted)

# Save combined table
df_all.to_csv("outputs/policy_comparison.csv", index=False)