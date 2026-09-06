import pandas as pd
import numpy as np

WINDOW = 50
PAD_VALUE = -1.0

std = pd.read_csv("data_cleaned/standard_cleaned.csv")
std = std.sort_values(['user_id', 'time_ms']).reset_index(drop=True)

group_sizes = std.groupby(['user_id', 'time_ms']).size()
print("=== 同一时间点记录数分布 ===")
print(group_sizes.describe())
print(f"最大值: {group_sizes.max()}")

MAX_GROUP = 15
n_truncated = (group_sizes > MAX_GROUP).sum()
print(f"组大小超过{MAX_GROUP}被截断的时间点数: {n_truncated}")
print(f"\n采用内层集合大小: {MAX_GROUP}")

grouped_lists = std.groupby(['user_id', 'time_ms'])['long_view'].apply(list).reset_index()
grouped_lists.columns = ['user_id', 'time_ms', 'lv_list']
grouped_lists = grouped_lists.sort_values(['user_id', 'time_ms']).reset_index(drop=True)

def pad_group(lst, size):
    arr = np.full(size, PAD_VALUE, dtype=np.int8)
    arr[:min(len(lst), size)] = lst[:size]
    return arr

all_sequences = []
all_keys = []

for uid, group in grouped_lists.groupby('user_id'):
    n = len(group)
    padded_groups = [pad_group(lst, MAX_GROUP) for lst in group['lv_list']]
    empty_group = np.full(MAX_GROUP, PAD_VALUE, dtype=np.int8)
    full_history = [empty_group] * WINDOW + padded_groups
    for i in range(n):
        seq = np.stack(full_history[i:i + WINDOW])
        all_sequences.append(seq)
    all_keys.append(group[['user_id', 'time_ms']])

sequences = np.array(all_sequences, dtype=np.int8)
keys = pd.concat(all_keys, ignore_index=True)

print(f"\n序列数组形状: {sequences.shape}（应为 [总时间点数, {WINDOW}, {MAX_GROUP}]）")
print(f"占用内存估计: {sequences.nbytes / 1024 / 1024:.1f} MB")

print("\n=== 核对：每个用户最早的时间点，序列应全部是 -1 ===")
first_idx = keys.groupby('user_id').head(1).index
bad = (sequences[first_idx] != PAD_VALUE).any(axis=(1, 2)).sum()
print(f"不满足的时间点数（应为0）: {bad}")

print("\n=== 核对：每个用户第二个时间点，序列第一个位置应等于第一个时间点的记录集合 ===")
second_idx = keys.groupby('user_id').nth(1).index
first_group_lists = grouped_lists.groupby('user_id')['lv_list'].first()
mismatch = 0
checked = 0
for idx in second_idx[:200]:
    uid = keys.loc[idx, 'user_id']
    step_values = sequences[idx][WINDOW - 1]
    actual_set = sorted(v for v in step_values if v != PAD_VALUE)
    expected_set = sorted(first_group_lists[uid][:MAX_GROUP])
    checked += 1
    if actual_set != expected_set:
        mismatch += 1
print(f"抽查{checked}个用户，不一致数（应为0）: {mismatch}")

np.save("data_cleaned/nested_sequences.npy", sequences)
keys.to_csv("data_cleaned/sequence_index.csv", index=False)
print("\n已保存 nested_sequences.npy 和 sequence_index.csv")
