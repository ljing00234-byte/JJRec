import pandas as pd
import os

DATA = "/Users/jj/Downloads/GOAI agent/JKRec/KuaiRand-1K/data"

std1 = pd.read_csv(f"{DATA}/log_standard_4_08_to_4_21_1k.csv")
std2 = pd.read_csv(f"{DATA}/log_standard_4_22_to_5_08_1k.csv")
std = pd.concat([std1, std2], ignore_index=True)

original_count = len(std)
print(f"原始行数: {original_count}")

dup_count = std.duplicated().sum()
std = std.drop_duplicates()
print(f"规则1 - 完全重复行数（剔除）: {dup_count}")

binary_cols = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
               'is_hate', 'is_rand', 'is_profile_enter', 'long_view']
bad_binary_mask = pd.Series(False, index=std.index)
for col in binary_cols:
    bad_binary_mask |= ~std[col].isin([0, 1])
binary_bad_count = bad_binary_mask.sum()
std = std[~bad_binary_mask]
print(f"规则2 - 二元字段异常行数（剔除）: {binary_bad_count}")

quarantine_mask = std['duration_ms'] <= 0
quarantined = std[quarantine_mask].copy()
quarantined['quarantine_reason'] = 'DURATION_MS_ZERO_LIKELY_REMOVED_VIDEO'
quarantine_count = quarantine_mask.sum()
std = std[~quarantine_mask]
print(f"规则3（调整为隔离）- duration_ms<=0 行数（隔离，不剔除）: {quarantine_count}")

final_count = len(std)
print(f"\nSilver层最终行数: {final_count}")
print(f"核对: {final_count} + {dup_count} + {binary_bad_count} + {quarantine_count} = "
      f"{final_count + dup_count + binary_bad_count + quarantine_count}（应等于原始行数 {original_count}）")

os.makedirs("data_cleaned", exist_ok=True)
std.to_csv("data_cleaned/standard_cleaned.csv", index=False)
quarantined.to_csv("data_cleaned/quarantined_removed_videos.csv", index=False)

print(f"\n主数据（Silver层）已保存到 data_cleaned/standard_cleaned.csv（{len(std)} 行）")
print(f"隔离数据已保存到 data_cleaned/quarantined_removed_videos.csv（{len(quarantined)} 行）")
