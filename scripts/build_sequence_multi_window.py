import pandas as pd
import numpy as np

PAD_VALUE = -1

std = pd.read_csv("data_cleaned/standard_cleaned.csv")
std = std.sort_values(['user_id', 'time_ms']).reset_index(drop=True)

grouped_lists = std.groupby(['user_id', 'time_ms'])['long_view'].apply(list).reset_index()
grouped_lists.columns = ['user_id', 'time_ms', 'lv_list']
grouped_lists = grouped_lists.sort_values(['user_id', 'time_ms']).reset_index(drop=True)

MAX_GROUP = 15

def pad_group(lst, size):
    arr = np.full(size, PAD_VALUE, dtype=np.int8)
    take = lst[:size]
    arr[:len(take)] = take
    return arr

def build_for_window(window):
    all_sequences = []
    for uid, group in grouped_lists.groupby('user_id'):
        n = len(group)
        padded_groups = [pad_group(lst, MAX_GROUP) for lst in group['lv_list']]
        empty_group = np.full(MAX_GROUP, PAD_VALUE, dtype=np.int8)
        full_history = [empty_group] * window + padded_groups
        for i in range(n):
            seq = np.stack(full_history[i:i + window])
            all_sequences.append(seq)
    sequences = np.array(all_sequences, dtype=np.int8)
    print(f"窗口{window}: 形状={sequences.shape}, 内存={sequences.nbytes/1024/1024:.1f}MB")
    return sequences

seq_w20 = build_for_window(20)
np.save("data_cleaned/nested_sequences_w20.npy", seq_w20)

seq_w100 = build_for_window(100)
np.save("data_cleaned/nested_sequences_w100.npy", seq_w100)

print("窗口20和100的序列数组已保存")
