import pandas as pd

std = pd.read_csv("data_cleaned/standard_cleaned.csv")

train = std[std['date'] <= 20220421]
test = std[std['date'] >= 20220422]

print(f"Silver层总行数: {len(std)}")
print(f"训练集（4/8-4/21）行数: {len(train)} ({len(train)/len(std)*100:.2f}%)")
print(f"测试集（4/22-5/8）行数: {len(test)} ({len(test)/len(std)*100:.2f}%)")
print(f"核对: {len(train)} + {len(test)} = {len(train)+len(test)}（应等于 {len(std)}）")
print(f"训练集用户数: {train['user_id'].nunique()}")
print(f"测试集用户数: {test['user_id'].nunique()}")

train.to_csv("data_cleaned/train_4_08_to_4_21.csv", index=False)
test.to_csv("data_cleaned/test_4_22_to_5_08.csv", index=False)
print("\n训练集已保存到 data_cleaned/train_4_08_to_4_21.csv")
print("测试集已保存到 data_cleaned/test_4_22_to_5_08.csv")
