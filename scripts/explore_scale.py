import pandas as pd

DATA = "/Users/jj/Downloads/GOAI agent/JKRec/KuaiRand-1K/data"

std1 = pd.read_csv(f"{DATA}/log_standard_4_08_to_4_21_1k.csv")
std2 = pd.read_csv(f"{DATA}/log_standard_4_22_to_5_08_1k.csv")
rand = pd.read_csv(f"{DATA}/log_random_4_22_to_5_08_1k.csv")
std = pd.concat([std1, std2], ignore_index=True)

print("=== 规模 ===")
for name, df in [("standard_4_08_to_4_21", std1), ("standard_4_22_to_5_08", std2),
                 ("random_4_22_to_5_08", rand), ("standard 合并", std)]:
    print(f"{name}: rows={len(df):,} users={df['user_id'].nunique():,} videos={df['video_id'].nunique():,}")

print()
print("=== 每用户历史长度（合并后的 standard 日志）===")
per_user = std.groupby('user_id').size()
print(per_user.describe(percentiles=[.25, .5, .75, .9, .99]))

print()
print("=== 时间跨度（合并后的 standard 日志）===")
print("date 范围:", std['date'].min(), "到", std['date'].max())

print()
print("=== 标签占比（合并后的 standard 日志）===")
for col in ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate', 'long_view']:
    print(f"{col}: {std[col].mean()*100:.4f}%")

print()
print("=== 时序检查（抽样 3 个用户）===")
sample_users = std['user_id'].unique()[:3]
for uid in sample_users:
    sub = std[std['user_id'] == uid]
    order_a = sub.sort_values('time_ms').index.tolist()
    order_b = sub.sort_values(['date', 'hourmin']).index.tolist()
    print(f"user_id={uid}: 行数={len(sub)}, 覆盖天数={sub['date'].nunique()}, "
          f"time_ms排序与date+hourmin排序一致={order_a == order_b}")
