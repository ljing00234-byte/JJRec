import pandas as pd

std = pd.read_csv("data_cleaned/standard_cleaned.csv")
std = std.sort_values(['user_id', 'time_ms']).reset_index(drop=True)

ts_group = std.groupby(['user_id', 'time_ms'])['long_view'].agg(['sum', 'count']).reset_index()
ts_group = ts_group.sort_values(['user_id', 'time_ms'])

ts_group['cum_sum_incl'] = ts_group.groupby('user_id')['sum'].cumsum()
ts_group['cum_count_incl'] = ts_group.groupby('user_id')['count'].cumsum()

ts_group['hist_long_view_count'] = ts_group.groupby('user_id')['cum_sum_incl'].shift(1)
ts_group['hist_interaction_count'] = ts_group.groupby('user_id')['cum_count_incl'].shift(1)
ts_group['hist_long_view_count'] = ts_group['hist_long_view_count'].fillna(0)
ts_group['hist_interaction_count'] = ts_group['hist_interaction_count'].fillna(0)

ts_group['hist_long_view_rate'] = ts_group['hist_long_view_count'] / ts_group['hist_interaction_count']
ts_group.loc[ts_group['hist_interaction_count'] == 0, 'hist_long_view_rate'] = float('nan')

std = std.merge(
    ts_group[['user_id', 'time_ms', 'hist_long_view_count', 'hist_interaction_count', 'hist_long_view_rate']],
    on=['user_id', 'time_ms'], how='left'
)

print("=== 人工核对：抽样一个有同时间戳记录的用户 ===")
dup_users = std[std.duplicated(subset=['user_id', 'time_ms'], keep=False)]['user_id'].unique()
sample_uid = dup_users[0] if len(dup_users) > 0 else std['user_id'].iloc[0]
sample = std[std['user_id'] == sample_uid].head(10)
print(sample[['user_id', 'time_ms', 'long_view', 'hist_interaction_count',
              'hist_long_view_count', 'hist_long_view_rate']].to_string())

print()
print("=== 核对：每个用户最早时间点的所有行，历史特征是否都是空值 ===")
first_ts = std.groupby('user_id')['time_ms'].transform('min')
first_ts_rows = std[std['time_ms'] == first_ts]
n_not_nan = first_ts_rows['hist_long_view_rate'].notna().sum()
print(f"最早时间点历史特征不为空的行数（应为0）: {n_not_nan}")

print()
print("=== 核对：同一时间点内的多条记录，历史特征是否完全相同 ===")
n_inconsistent = std.groupby(['user_id', 'time_ms'])['hist_long_view_rate'].nunique()
n_bad = (n_inconsistent > 1).sum()
print(f"同一时间点内历史特征不一致的(用户,时间点)组合数（应为0）: {n_bad}")

train = std[std['date'] <= 20220421]
test = std[std['date'] >= 20220422]

print(f"\n训练集行数: {len(train)}, 测试集行数: {len(test)}")
print(f"训练集历史特征为空的行数: {train['hist_long_view_rate'].isna().sum()}")
print(f"测试集历史特征为空的行数（应该很小）: {test['hist_long_view_rate'].isna().sum()}")

train.to_csv("data_cleaned/model2_features_train.csv", index=False)
test.to_csv("data_cleaned/model2_features_test.csv", index=False)
print("\n已保存 model2_features_train.csv 和 model2_features_test.csv")
