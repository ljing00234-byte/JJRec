import pandas as pd
import os

DATA = os.environ.get("KUAIRAND_DATA_DIR")
if not DATA:
    raise RuntimeError("请设置环境变量 KUAIRAND_DATA_DIR，指向本地 KuaiRand-1K/data 目录")

std1 = pd.read_csv(f"{DATA}/log_standard_4_08_to_4_21_1k.csv")
std2 = pd.read_csv(f"{DATA}/log_standard_4_22_to_5_08_1k.csv")
std = pd.concat([std1, std2], ignore_index=True)
std = std.drop_duplicates()

video_basic = pd.read_csv(f"{DATA}/video_features_basic_1k.csv")

zero_dur = std[std['duration_ms'] <= 0]
print(f"duration_ms<=0 的行数: {len(zero_dur)} / {len(std)} ({len(zero_dur)/len(std)*100:.4f}%)")

print()
print("=== 这些行里，各信号为1(或>0)的比例 ===")
for col in ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view']:
    n = (zero_dur[col] == 1).sum()
    print(f"{col}=1: {n} 行 ({n/len(zero_dur)*100:.4f}%)")
n_play = (zero_dur['play_time_ms'] > 0).sum()
print(f"play_time_ms>0: {n_play} 行 ({n_play/len(zero_dur)*100:.4f}%)")

print()
print("=== 对照：全体数据里同样信号的比例 ===")
for col in ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view']:
    n = (std[col] == 1).sum()
    print(f"{col}=1: {n} 行 ({n/len(std)*100:.4f}%)")
n_play_all = (std['play_time_ms'] > 0).sum()
print(f"play_time_ms>0: {n_play_all} 行 ({n_play_all/len(std)*100:.4f}%)")

print()
print("=== 这些 duration_ms=0 涉及的视频，在 video_features_basic 里的情况 ===")
zero_dur_videos = zero_dur['video_id'].unique()
print(f"涉及的独立视频数: {len(zero_dur_videos)}")
vb_subset = video_basic[video_basic['video_id'].isin(zero_dur_videos)]
print(f"能在 video_features_basic 里找到的视频数: {len(vb_subset)}")
print(f"video_duration 缺失(NaN)的数量: {vb_subset['video_duration'].isna().sum()}")
print(f"video_duration 为0或负数的数量: {(vb_subset['video_duration'] <= 0).sum()}")
n_normal = (vb_subset['video_duration'] > 0).sum()
print(f"video_duration 是正常正数的数量: {n_normal}")
if n_normal > 0:
    print("\n样本（basic表里video_duration正常，但日志里duration_ms=0）：")
    normal_ones = vb_subset[vb_subset['video_duration'] > 0].head(5)
    print(normal_ones[['video_id', 'video_duration', 'video_type', 'visible_status']].to_string())
