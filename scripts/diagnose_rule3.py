import pandas as pd
import os

DATA = os.environ.get("KUAIRAND_DATA_DIR")
if not DATA:
    raise RuntimeError("请设置环境变量 KUAIRAND_DATA_DIR，指向本地 KuaiRand-1K/data 目录")

std1 = pd.read_csv(f"{DATA}/log_standard_4_08_to_4_21_1k.csv")
std2 = pd.read_csv(f"{DATA}/log_standard_4_22_to_5_08_1k.csv")
std = pd.concat([std1, std2], ignore_index=True)

std = std.drop_duplicates()
binary_cols = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
               'is_hate', 'is_rand', 'is_profile_enter', 'long_view']
bad_binary_mask = pd.Series(False, index=std.index)
for col in binary_cols:
    bad_binary_mask |= ~std[col].isin([0, 1])
std = std[~bad_binary_mask]

print(f"规则1、2清洗后剩余行数: {len(std)}")
print()

conditions = {
    "play_time_ms < 0": std['play_time_ms'] < 0,
    "duration_ms <= 0": std['duration_ms'] <= 0,
    "date 超出 [20220408, 20220508]": (std['date'] < 20220408) | (std['date'] > 20220508),
    "tab 超出 [0,14]": (std['tab'] < 0) | (std['tab'] > 14),
}

print("=== 各条件单独命中的行数 ===")
for name, mask in conditions.items():
    n = mask.sum()
    print(f"{name}: {n} 行 ({n/len(std)*100:.4f}%)")
    if n > 0:
        print(std.loc[mask].head(5).to_string())
    print()

combined = pd.Series(False, index=std.index)
for mask in conditions.values():
    combined |= mask
print(f"=== 合并后（去重）总命中行数: {combined.sum()} ===")

print()
print("=== 命中条件个数的分布（1行命中几个条件）===")
hit_count = sum(m.astype(int) for m in conditions.values())
print(hit_count[combined].value_counts().sort_index())
