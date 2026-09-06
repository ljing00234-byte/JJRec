import pandas as pd
import os

DATA = os.environ.get("KUAIRAND_DATA_DIR")
if not DATA:
    raise RuntimeError("请设置环境变量 KUAIRAND_DATA_DIR，指向本地 KuaiRand-1K/data 目录")

std1 = pd.read_csv(f"{DATA}/log_standard_4_08_to_4_21_1k.csv")
std2 = pd.read_csv(f"{DATA}/log_standard_4_22_to_5_08_1k.csv")
std = pd.concat([std1, std2], ignore_index=True)

std['dt_utc'] = pd.to_datetime(std['time_ms'], unit='ms')
std['dt'] = std['dt_utc'] + pd.Timedelta(hours=8)  # 修正为北京时间 UTC+8

range_start = pd.Timestamp('2022-04-08')
range_end = pd.Timestamp('2022-05-09')  # 含5月8日全天

print("=== 时区修正后：time_ms 换算日期是否落在声称范围内 ===")
out_of_range = (std['dt'] < range_start) | (std['dt'] >= range_end)
print(f"超出范围的行数: {out_of_range.sum()} / {len(std)} ({out_of_range.sum()/len(std)*100:.4f}%)")
if out_of_range.sum() > 0:
    print(std.loc[out_of_range, ['user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'dt_utc', 'dt']].head(10).to_string())
else:
    print("时区修正后，全部行都落在声称的日期范围内。")

print()
print("=== 相邻间隔检查（不受时区影响，沿用之前结果，本次不重跑）===")
print("此前结果：5个抽样用户，负数间隔=0，超过7天的跳跃=0")
