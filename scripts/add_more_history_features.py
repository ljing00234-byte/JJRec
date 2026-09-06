import pandas as pd

train = pd.read_csv("data_cleaned/model2_features_train.csv")
test = pd.read_csv("data_cleaned/model2_features_test.csv")
full = pd.concat([train, test], ignore_index=True)
full = full.sort_values(['user_id', 'time_ms']).reset_index(drop=True)

signals = ['is_like', 'is_comment', 'is_forward', 'is_profile_enter']

agg_dict = {s: (s, 'sum') for s in signals}
ts_group = full.groupby(['user_id', 'time_ms']).agg(**agg_dict, row_count=('long_view', 'size')).reset_index()
ts_group = ts_group.sort_values(['user_id', 'time_ms'])

ts_group['cum_count_incl'] = ts_group.groupby('user_id')['row_count'].cumsum()
ts_group['hist_count_check'] = ts_group.groupby('user_id')['cum_count_incl'].shift(1).fillna(0)

for s in signals:
    ts_group[f'cum_{s}_incl'] = ts_group.groupby('user_id')[s].cumsum()
    hist_count = ts_group.groupby('user_id')[f'cum_{s}_incl'].shift(1).fillna(0)
    ts_group[f'hist_{s}_rate'] = hist_count / ts_group['hist_count_check']
    ts_group.loc[ts_group['hist_count_check'] == 0, f'hist_{s}_rate'] = float('nan')

new_cols = [f'hist_{s}_rate' for s in signals]
merge_cols = ['user_id', 'time_ms', 'hist_count_check'] + new_cols
full = full.merge(ts_group[merge_cols], on=['user_id', 'time_ms'], how='left')

print("=== 复核：重新算的互动次数应等于已有的 hist_interaction_count ===")
mismatch = (full['hist_interaction_count'] != full['hist_count_check']).sum()
print(f"不一致的行数（应为0）: {mismatch}")
full = full.drop(columns=['hist_count_check'])

print("\n=== 每个新增历史特征的空值行数（应与 hist_long_view_rate 一致）===")
print(f"hist_long_view_rate 空值数: {full['hist_long_view_rate'].isna().sum()}")
for c in new_cols:
    print(f"{c} 空值数: {full[c].isna().sum()}")

train_new = full[full['date'] <= 20220421]
test_new = full[full['date'] >= 20220422]

train_new.to_csv("data_cleaned/model2_features_train.csv", index=False)
test_new.to_csv("data_cleaned/model2_features_test.csv", index=False)
print("\n已更新 model2_features_train.csv 和 model2_features_test.csv，新增4个历史比率特征")
