import pandas as pd

std = pd.read_csv("data_cleaned/standard_cleaned.csv")
group_sizes = std.groupby(['user_id', 'time_ms']).size()

total = len(group_sizes)
print(f"总时间点数: {total}")

for cap in [10, 12, 15]:
    n_exceed = (group_sizes > cap).sum()
    print(f"组大小超过{cap}的时间点数: {n_exceed} ({n_exceed/total*100:.4f}%)")

print(f"\n95分位: {group_sizes.quantile(0.95)}")
print(f"99分位: {group_sizes.quantile(0.99)}")
print(f"99.9分位: {group_sizes.quantile(0.999)}")
