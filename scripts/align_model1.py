import pandas as pd

train1 = pd.read_csv("data_cleaned/gold_model1_train.csv")
test1 = pd.read_csv("data_cleaned/gold_model1_test.csv")

train2 = pd.read_csv("data_cleaned/gold_model2_train.csv")
test2 = pd.read_csv("data_cleaned/gold_model2_test.csv")

key_cols = ['user_id', 'video_id', 'time_ms']


def filter_common(df1, df2, name):
    before = len(df1)
    keys2 = df2[key_cols].drop_duplicates()
    filtered = df1.merge(keys2, on=key_cols, how='inner')
    print(f"{name}: model1原始行数={before}, 过滤后行数={len(filtered)}, 剔除={before - len(filtered)}")
    return filtered


train1_filtered = filter_common(train1, train2, "train")
test1_filtered = filter_common(test1, test2, "test")

print(f"\n对齐后 train1 行数 vs model2 train 行数: {len(train1_filtered)} vs {len(train2)}")
print(f"对齐后 test1 行数 vs model2 test 行数: {len(test1_filtered)} vs {len(test2)}")

train1_filtered.to_csv("data_cleaned/gold_model1_train_aligned.csv", index=False)
test1_filtered.to_csv("data_cleaned/gold_model1_test_aligned.csv", index=False)
print("\n对齐后的 model1 训练/测试表已保存")
