import pandas as pd
import numpy as np

seq_index = pd.read_csv("data_cleaned/sequence_index.csv")
seq_index['seq_idx'] = np.arange(len(seq_index))

train = pd.read_csv("data_cleaned/train_4_08_to_4_21.csv")
test = pd.read_csv("data_cleaned/test_4_22_to_5_08.csv")

def build_mapping(df, name):
    merged = df.merge(seq_index[['user_id', 'time_ms', 'seq_idx']], on=['user_id', 'time_ms'], how='left')
    n_missing = merged['seq_idx'].isna().sum()
    print(f"{name}: 总行数={len(df)}, 匹配到序列编号的行数={len(merged)-n_missing}, 未匹配到的行数={n_missing}")
    return merged

train_merged = build_mapping(train, "train")
test_merged = build_mapping(test, "test")

train_seq_idx = train_merged['seq_idx'].values.astype(np.int32)
test_seq_idx = test_merged['seq_idx'].values.astype(np.int32)
train_labels = train_merged['long_view'].values.astype(np.int8)
test_labels = test_merged['long_view'].values.astype(np.int8)

print(f"\ntrain 正例比例: {train_labels.mean()*100:.4f}%")
print(f"test 正例比例: {test_labels.mean()*100:.4f}%")

print("\n=== 抽样核对：映射的序列编号是否对应正确的 user_id/time_ms ===")
sequences_meta = seq_index.set_index('seq_idx')
sample = train_merged.sample(5, random_state=42)
all_match = True
for idx, row in sample.iterrows():
    seq_idx = int(row['seq_idx'])
    meta = sequences_meta.loc[seq_idx]
    match = (meta['user_id'] == row['user_id']) and (meta['time_ms'] == row['time_ms'])
    if not match:
        all_match = False
    print(f"行 user_id={row['user_id']}, time_ms={row['time_ms']} -> "
          f"序列编号={seq_idx} -> 查回 user_id={meta['user_id']}, time_ms={meta['time_ms']}, 一致={match}")
print(f"\n全部抽样一致: {all_match}")

np.save("data_cleaned/train_seq_idx.npy", train_seq_idx)
np.save("data_cleaned/test_seq_idx.npy", test_seq_idx)
np.save("data_cleaned/train_labels.npy", train_labels)
np.save("data_cleaned/test_labels.npy", test_labels)
print("\n映射和标签文件已保存")
